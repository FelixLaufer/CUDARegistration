//----------------------------------------------------------------------
/*!\file    gpu_algorithms/templateRegistration.cu
 *
 * \author  Felix Laufer
 *
 *
 * CUDA: Fast rotation-invariant template registration on large 2D matrices
 *
 */
//----------------------------------------------------------------------

#include <cufft.h>
#include "gpu_algorithms/phaseCorrelation.cu"
#include "gpu_algorithms/debugPrint.cu"
#include <stdio.h>

namespace gpu_algorithms
{
namespace cuda
{
namespace template_registration
{

//----------------------------------------------------------------------
// Kernel functions
//----------------------------------------------------------------------

// Multiplication of a complex signal a's magnitude with another unchanged complex signal b. Optionally shift the output.
template<bool param_inverse_shift>
static __global__ void ComplexPointwiseMagnitudeMulAndScale(const Complex* a, const Complex* b, Complex* out, const unsigned int stream_size, const unsigned int matrix_size, const float normalization_factor, const bool allow_highpass)
{
    const unsigned int numThreads = blockDim.x * gridDim.x;
    const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = threadID; i < stream_size; i += numThreads)
    {
    	Complex magnitude = (Complex) {sqrtf(a[i].x * a[i].x + a[i].y * a[i].y), 0.0f};
    	Complex product = (allow_highpass) ? ComplexMul(magnitude, b[i]) : magnitude;

    	unsigned int index = i;
    	if (param_inverse_shift)
    	{
			int y = i / matrix_size;
			int x = i - y * matrix_size;
			index  = SequentialIndex2DInverseFFTShift(x, y, matrix_size);
    	}
        out[index] = (Complex) {product.x / normalization_factor, product.y / normalization_factor};
    }
}

// Calculate a data stream of complex point-wise mean squared errors of the given input streams
static __global__ void ComplexPointwiseMeanSquaredError(const Complex *a, const Complex *b, Complex *out, const unsigned int stream_size)
{
    const unsigned int numThreads = blockDim.x * gridDim.x;
    const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = threadID; i < stream_size; i += numThreads)
    {
    	Complex difference = (Complex) {a[i].x - b[i].x, a[i].y - b[i].y};
    	Complex difference_squared = ComplexMul(difference, difference);
        out[i] = difference_squared;
    }
}

// Transformation of a complex cartesian matrix to polar space. Optionally zero-pad and shift the output.
static __global__ void Cartesian2PolarTransform(const Complex *idata, Complex *odata, const unsigned int rho_theta_matrix_stream_size, const unsigned int matrix_size, const unsigned int rho_size, const unsigned int theta_size)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	const unsigned int radius = (matrix_size - 1) / 2 + 1;
	const float step_rho = (float) radius / rho_size;
	//const float step_rho = sqrtf(2.0f * radius * radius) / rho_size; // TODO: Which one is better?
	const float step_theta = 1.0f * M_PI / theta_size;

	for (unsigned int i = threadID; i < rho_theta_matrix_stream_size; i += numThreads)
	{
	    const unsigned int theta_n = i / rho_size;
		const unsigned int rho_n = i - theta_n * rho_size;

		Real data;

		if (rho_n >= rho_size || theta_n >= theta_size)
		{
			data = 0.0f;
		}
		else
		{
			const float rho = rho_n * step_rho;
			const float theta = theta_n * step_theta;

			float x = rho * cos(theta) + (matrix_size - 1) / 2;
			float y = rho * sin(theta) + (matrix_size - 1) / 2;

			y = (float)matrix_size - 1.0f - y;

			data = BilinearInterpolation(x, y, idata, matrix_size);
		}

		odata[i].x = data;
		odata[i].y = 0.0f;
	}
}

// Real to Complex with optional circular shift and optional weighting
template<bool param_shift, bool param_weighted>
static __global__ void Real2ComplexPadAndShift(const Real *idata, Complex *odata, const unsigned int size, const unsigned int matrix_size, const unsigned int matrix_size_expanded, const Real *weights)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	const unsigned int o_i_block_offset = (matrix_size_expanded - 1) / 2 - (matrix_size - 1) / 2;

	for (unsigned int i = threadID; i < size; i += numThreads)
	{
		int o_block_y = i / matrix_size_expanded;
		int o_block_x = i - o_block_y * matrix_size_expanded;

		const int i_block_x = o_block_x - o_i_block_offset;
		const int i_block_y = o_block_y - o_i_block_offset;

		Real data;
		if(!(0 <= i_block_x && i_block_x < matrix_size && 0 <= i_block_y && i_block_y < matrix_size))
		{
			data = 0.0f;
		}
		else
		{
			const int i_matrix_x = i_block_x;
			const int i_matrix_y = i_block_y;
			Real weight = param_weighted ? weights[o_block_y * matrix_size_expanded + o_block_x] : 1.0f;
			const bool is_valid_coordinate = (0 <= i_matrix_x && i_matrix_x < matrix_size && 0 <= i_matrix_y && i_matrix_y < matrix_size);
			data = is_valid_coordinate ? idata[i_matrix_y * matrix_size + i_matrix_x] * weight: 0.0f;
		}

		unsigned int index = i;

		if (param_shift)
		{
			index = SequentialIndex2DFFTShift(o_block_x, o_block_y, matrix_size_expanded);
		}

		odata[index].x = data;
		odata[index].y = 0.0f;
	}
}

// Generate a high pass kernel in time domain
template<bool param_shift>
static __global__ void HighPassKernel(Complex *odata, const unsigned int size, const unsigned int matrix_size, const float sigma)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	const unsigned int offset = (matrix_size - 1) / 2;

	for (unsigned int i = threadID; i < size; i += numThreads)
	{
		const unsigned int y = i / matrix_size;
		const unsigned int x = i - y * matrix_size;

		const int x_o = x - offset;
		const int y_o = y - offset;

		float s = 2 * sigma * sigma;
		float gaussian_lowpass = 1.0f / (M_PI * s) * (expf(-(x_o * x_o + y_o * y_o) / s));
		float gaussian_highpass = (x_o == 0 && y_o == 0) ? 2.0f - gaussian_lowpass : -gaussian_lowpass;

		unsigned int index = i;

		if (param_shift)
		{
			index = SequentialIndex2DFFTShift(x, y, matrix_size);
		}

		odata[index] = (Complex) {gaussian_highpass, 0.0f};
	}
}

//----------------------------------------------------------------------
// Host functions
//----------------------------------------------------------------------

static __host__ float MeanSquaredError(const Complex *a, const Complex *b, const unsigned int nx, const unsigned int ny)
{
	const unsigned int max_stream_threads_per_block = 256;

	const unsigned int frame_stream_size = nx * ny;
	const unsigned int stream_threads_per_block = min(max_stream_threads_per_block, frame_stream_size);
	const unsigned int stream_blocks = ceil(frame_stream_size / (float) stream_threads_per_block);

	Complex *errors;
	cudaMalloc((void**)&errors, frame_stream_size * sizeof(Complex));

	const dim3 grid(ceil(frame_stream_size / (float) stream_threads_per_block));
	const dim3 block(stream_threads_per_block);

	// Calculate point-wise errors
	ComplexPointwiseMeanSquaredError
	<<<grid, block>>>
	(
		a, b, errors, frame_stream_size
	);

	// Sum up point-wise errors
	Complex squared_mean_error = (Complex) {0.0f, 0.0f};
	SumReduce(errors, &squared_mean_error, frame_stream_size);

	cudaFree(errors);

	return squared_mean_error.x;
}


template <bool param_rotation_allowed>
static __host__ void TanslationRotationEstimation(const Real *iframe_a_data, const Real *iframe_b_data, const unsigned int frame_a_nx, const unsigned int frame_b_nx, const bool allow_highpass_filtering = true, const unsigned int max_degree_resolution = 180)
{
	// Computation threads per block for 1d data streams
	const unsigned int stream_threads_per_block = 256;

	const unsigned int frame_a_stream_size = frame_a_nx * frame_a_nx;
	const unsigned int frame_b_stream_size = frame_b_nx * frame_b_nx;
	const unsigned int nx = frame_a_nx;
	const unsigned int frame_stream_size = frame_a_stream_size;

	Real *frame_a_data,
		 *frame_b_data;

	Complex *frame_a_complex,
			*frame_b_complex;

	cudaMalloc((void**)&frame_a_data, (frame_a_stream_size + frame_b_stream_size) * sizeof(Complex));
	cudaMalloc((void**)&frame_a_complex, frame_a_stream_size * sizeof(Complex) * 2);

	frame_b_data = &frame_a_data[frame_a_stream_size];
	frame_b_complex = &frame_a_complex[frame_a_stream_size];

	// Prepare grid, block and shared memory configuration for block matrix extraction
	const dim3 k0_grid(ceil(frame_stream_size / (float) stream_threads_per_block));
	const dim3 k0_block(stream_threads_per_block);

	// Transfer input data to device memory
	cudaMemcpy(frame_a_data, iframe_a_data, frame_a_stream_size * sizeof(Real), cudaMemcpyHostToDevice);
	cudaMemcpy(frame_b_data, iframe_b_data, frame_b_stream_size * sizeof(Real), cudaMemcpyHostToDevice);

	// Expand and pad frame a
	Real2ComplexPadAndShift<false, false>
	<<<k0_grid, k0_block>>>
	(
		frame_a_data, frame_a_complex,
		frame_a_stream_size,
		frame_a_nx, frame_a_nx,
		NULL
	);

	// Expand and pad shift frame b
	Real2ComplexPadAndShift<false, false>
	<<<k0_grid, k0_block>>>
	(
		frame_b_data, frame_b_complex,
		frame_a_stream_size,
		frame_b_nx, frame_a_nx,
		NULL
	);

	float rotation_angle = 0.0f;
	float corrected_mean_squared_error = 0.0f;
	Vec2f translation_vector = (Vec2f) {0.0f, 0.0f};

	if (!param_rotation_allowed)
	{
		translation_vector = phase_correlation::TranslationVector<false>(frame_a_complex, frame_b_complex, nx);
		corrected_mean_squared_error = MeanSquaredError(frame_b_complex, frame_a_complex, nx, nx);
	}
	else
	{
		const unsigned int frame_polar_matrix_size_rho = (sqrt(2 * ((nx - 1) / 2 + 1) * ((nx - 1) / 2 + 1)));
		const unsigned int frame_polar_matrix_size_theta = min(((2 * nx) / 4 * 4) , max_degree_resolution);
		const unsigned int frame_stream_size = nx * nx;
		const unsigned int frame_stream_size_polar = frame_polar_matrix_size_rho * frame_polar_matrix_size_theta;

		Complex *frame_a_data_complex,
				*frame_b_data_complex,
				*frame_a_data_complex_filtered,
				*frame_b_data_complex_filtered,
				*highpass_kernel_complex,
				*frame_a_data_polar_complex,
				*frame_b_data_polar_complex;

		cudaMalloc((void**)&frame_a_data_complex, (5 * frame_stream_size + 2 * frame_stream_size_polar) * sizeof(Complex));

		frame_b_data_complex = &frame_a_data_complex[frame_stream_size];
		frame_a_data_complex_filtered = &frame_b_data_complex[frame_stream_size];
		frame_b_data_complex_filtered = &frame_a_data_complex_filtered[frame_stream_size];
		highpass_kernel_complex = &frame_b_data_complex_filtered[frame_stream_size];
		frame_a_data_polar_complex = &highpass_kernel_complex[frame_stream_size_polar];
		frame_b_data_polar_complex = &frame_a_data_polar_complex[frame_stream_size_polar];

		// Prepare 1D FFT C2C batched plans
		cufftHandle plan_1d_complex_row, plan_1d_complex_col;
		{
			int n_row[] = {nx};
			int n_col[] = {nx};
			int inembed_row[] = {nx};
			int onembed_row[] = {nx};
			int inembed_col[] = {1};
			int onembed_col[] = {1};
			cufftPlanMany(&plan_1d_complex_row, 1, n_row, inembed_row, 1, nx, onembed_row, 1, nx, CUFFT_C2C, nx);
			cufftPlanMany(&plan_1d_complex_col, 1, n_col, inembed_col, nx, 1, onembed_col, nx, 1, CUFFT_C2C, nx);
		}

		// Prepare grid and block configuration for polar transformations
		const dim3 k1_grid(ceil(frame_stream_size_polar / (float) stream_threads_per_block));
		const dim3 k1_block(stream_threads_per_block);

		// Generate gaussian high pass filter kernel
		HighPassKernel<true>
		<<<k0_grid, k0_block>>>
		(
			highpass_kernel_complex, frame_stream_size, nx, 0.3f
		);

		// FFT both frames first row-wise then column-wise
		cufftExecC2C(plan_1d_complex_row, frame_a_complex, frame_a_data_complex, CUFFT_FORWARD);
		cufftExecC2C(plan_1d_complex_col, frame_a_data_complex, frame_a_data_complex, CUFFT_FORWARD);
		cufftExecC2C(plan_1d_complex_row, frame_b_complex, frame_b_data_complex, CUFFT_FORWARD);
		cufftExecC2C(plan_1d_complex_col, frame_b_data_complex, frame_b_data_complex, CUFFT_FORWARD);
		cufftExecC2C(plan_1d_complex_row, highpass_kernel_complex, highpass_kernel_complex, CUFFT_FORWARD);
		cufftExecC2C(plan_1d_complex_col, highpass_kernel_complex, highpass_kernel_complex, CUFFT_FORWARD);

		cufftDestroy(plan_1d_complex_row);
		cufftDestroy(plan_1d_complex_col);

		// High pass filter both frame's magnitudes
		ComplexPointwiseMagnitudeMulAndScale<true>
		<<<k0_grid, k0_block>>>
		(
			frame_a_data_complex, highpass_kernel_complex, frame_a_data_complex_filtered,
			frame_stream_size, nx, frame_stream_size,
			allow_highpass_filtering
		);

		ComplexPointwiseMagnitudeMulAndScale<true>
		<<<k0_grid, k0_block>>>
		(
			frame_b_data_complex, highpass_kernel_complex, frame_b_data_complex_filtered,
			frame_stream_size, nx, frame_stream_size,
			allow_highpass_filtering
		);


		// Transform both frames FFT coefficients to polar space
		Cartesian2PolarTransform
		<<<k1_grid, k1_block>>>
		(
			frame_a_data_complex_filtered, frame_a_data_polar_complex,
			frame_stream_size_polar, nx,
			frame_polar_matrix_size_rho,
			frame_polar_matrix_size_theta
		);

		Cartesian2PolarTransform
		<<<k1_grid, k1_block>>>
		(
			frame_b_data_complex_filtered, frame_b_data_polar_complex,
			frame_stream_size_polar, nx,
			frame_polar_matrix_size_rho,
			frame_polar_matrix_size_theta
		);

		// Correlate polar frames and calculate estimated rotation
		// Note: Phase correlation cannot distinguish between angle and angle + 180 degree => try both and measure errors
		const unsigned int peak_index_rotation = phase_correlation::PeakIndex<true, false>(frame_a_data_polar_complex, frame_b_data_polar_complex, frame_polar_matrix_size_rho, frame_polar_matrix_size_theta);
		float base_rotation = M_PI * ((float) peak_index_rotation / frame_polar_matrix_size_rho) / frame_polar_matrix_size_theta;

		float rotation_angle_1 = base_rotation;
		Rotate
		<<<k0_grid, k0_block>>>
		(
			frame_b_complex, frame_a_data_complex_filtered,
			frame_stream_size,
			nx, rotation_angle_1
		);

		float rotation_angle_2 = base_rotation + M_PI;
		Rotate
		<<<k0_grid, k0_block>>>
		(
			frame_b_complex, frame_b_data_complex_filtered,
			frame_stream_size,
			nx, rotation_angle_2
		);

		Vec2f translation_vector_1 = phase_correlation::TranslationVector<false>(frame_a_complex, frame_a_data_complex_filtered, nx);
		Translate
		<<<k0_grid, k0_block>>>
		(
			frame_a_data_complex_filtered, frame_b_complex,
			frame_stream_size,
			nx,
			(int) round(translation_vector_1.x), (int) round(translation_vector_1.y)
		);
		const float mean_squared_error_1 = MeanSquaredError(frame_b_complex, frame_a_complex, nx, nx);

		Vec2f translation_vector_2 = phase_correlation::TranslationVector<false>(frame_a_complex, frame_b_data_complex_filtered, nx);
		Translate
		<<<k0_grid, k0_block>>>
		(
			frame_b_data_complex_filtered, frame_b_complex,
			frame_stream_size,
			nx,
			(int) round(translation_vector_2.x), (int) round(translation_vector_2.y)
		);
		const float mean_squared_error_2 = MeanSquaredError(frame_b_complex, frame_a_complex, nx, nx);

		if (mean_squared_error_1 < mean_squared_error_2)
		{
			rotation_angle = rotation_angle_1;
			translation_vector = translation_vector_1;
			corrected_mean_squared_error = mean_squared_error_1;
		}
		else
		{
			rotation_angle = rotation_angle_2;
			translation_vector = translation_vector_2;
			corrected_mean_squared_error = mean_squared_error_2;
		}
	}

	printf("Rotation: %4.2f° \n", rotation_angle * 180.0f / M_PI);
	printf("Translation: (%0.0f, %0.0f) \n", translation_vector.x, translation_vector.y);
	printf("Remaining error: %4.4f \n", corrected_mean_squared_error);
}


__host__ void TemplateRegistration(const float* iframe_a_data, const float* iframe_b_data, float* result_frame, const unsigned int frame_a_matrix_size, const unsigned int frame_b_matrix_size, bool weighting_window, bool rotation_allowed)
{
	TanslationRotationEstimation<true>(iframe_a_data, iframe_b_data, frame_a_matrix_size, frame_b_matrix_size);
}


}
}
}
