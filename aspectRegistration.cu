#include <cufft.h>
#include "projects/aspect_maps_memory/gpu_algorithms/phaseCorrelation.cu"
#include "projects/aspect_maps_memory/gpu_algorithms/debugPrint.cu"
#include <stdio.h>


namespace finroc
{
namespace aspect_maps_memory
{
namespace gpu_algorithms
{
namespace cuda
{
namespace aspect_registration
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
static __host__ void RotationEstimation(const Real *iframe_a_data, const Real *iframe_b_data, const unsigned int frame_a_nx, const unsigned int frame_b_nx, const bool allow_highpass_filtering = true, const unsigned int max_degree_resolution = 180)
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



//__host__ void SubAspectRegistration_BLA(const float* iframe_a_data, const float* iframe_b_data, float* result_frame, const unsigned int frame_a_matrix_size, const unsigned int frame_b_matrix_size, bool weighting_window, bool rotation_allowed)
//{
//	// Return immediately in case of wrong size specifications
//	if (frame_b_matrix_size > frame_a_matrix_size)
//	{
//		return;
//	}
//
//	const unsigned int stream_threads_per_block = 256;
//	const unsigned int max_degree_resolution = 180;
//
//	const unsigned int frame_polar_matrix_size_rho = (sqrt(2 * ((frame_a_matrix_size - 1) / 2 + 1) * ((frame_a_matrix_size - 1) / 2 + 1)));
//	const unsigned int frame_polar_matrix_size_theta = min(((2 * frame_a_matrix_size) / 4 * 4) , max_degree_resolution);
//
//	// Stream sizes of raw frame data
//	const unsigned int frame_a_stream_size = frame_a_matrix_size * frame_a_matrix_size;
//	const unsigned int frame_b_stream_size = frame_b_matrix_size * frame_b_matrix_size;
//	const unsigned int frame_stream_size_polar = frame_polar_matrix_size_rho * frame_polar_matrix_size_theta;
//
//	// Actual byte sizes of raw frame real and complex data
//	const unsigned int frame_a_stream_size_real = frame_a_stream_size * sizeof(Real);
//	const unsigned int frame_b_stream_size_real = frame_b_stream_size * sizeof(Real);
//	const unsigned int frame_stream_size_complex = frame_a_stream_size * sizeof(Complex);
//	const unsigned int frame_stream_size_polar_complex = frame_stream_size_polar * sizeof(Complex);
//
//	// Allocate all device memory
//	Real *frame_data,
//		 *frame_a_data,
//		 *frame_b_data;
//
//	Complex *frame_data_complex,
//			*frame_a_data_complex,
//			*frame_b_data_complex,
//			*frame_a_data_complex_filtered,
//			*frame_b_data_complex_filtered,
//			*frame_a_data_complex_copy,
//			*frame_b_data_complex_copy,
//			*frame_data_complex_polar,
//			*frame_a_data_complex_polar,
//			*frame_b_data_complex_polar,
//			*highpass_kernel_complex;
//
//	int *max_indices;
//
//	Complex *squared_mean_error;
//
//	cudaMalloc((void**)&frame_data, frame_a_stream_size_real + frame_b_stream_size_real);
//	cudaMalloc((void**)&frame_data_complex, frame_stream_size_complex * 6);
//	cudaMalloc((void**)&frame_data_complex_polar, frame_stream_size_polar_complex * 2);
//	cudaMalloc((void**)&highpass_kernel_complex, frame_stream_size_complex);
//	cudaMalloc((void**)&max_indices, 1 * sizeof(int));
//	cudaMalloc((void**)&squared_mean_error, 2 * sizeof(Complex));
//
//	frame_a_data = &frame_data[0];
//	frame_b_data = &frame_a_data[frame_a_stream_size];
//	frame_a_data_complex = &frame_data_complex[0];
//	frame_b_data_complex = &frame_a_data_complex[frame_a_stream_size];
//	frame_a_data_complex_filtered = &frame_b_data_complex[frame_a_stream_size];
//	frame_b_data_complex_filtered = &frame_a_data_complex_filtered[frame_a_stream_size];
//	frame_a_data_complex_copy = &frame_b_data_complex_filtered[frame_a_stream_size];
//	frame_b_data_complex_copy = &frame_a_data_complex_copy[frame_a_stream_size];
//	frame_a_data_complex_polar = &frame_data_complex_polar[0];
//	frame_b_data_complex_polar = &frame_a_data_complex_polar[frame_stream_size_polar];
//
//	// Prepare 1D FFT C2C batched plans
//	cufftHandle plan_1d_complex_row, plan_1d_complex_col;
//	{
//		int n_row[] = {frame_a_matrix_size};
//		int n_col[] = {frame_a_matrix_size};
//		int inembed_row[] = {frame_a_matrix_size};
//		int onembed_row[] = {frame_a_matrix_size};
//		int inembed_col[] = {1};
//		int onembed_col[] = {1};
//		cufftPlanMany(&plan_1d_complex_row, 1, n_row, inembed_row, 1, frame_a_matrix_size, onembed_row, 1, frame_a_matrix_size, CUFFT_C2C, frame_a_matrix_size);
//		cufftPlanMany(&plan_1d_complex_col, 1, n_col, inembed_col, frame_a_matrix_size, 1, onembed_col, frame_a_matrix_size, 1, CUFFT_C2C, frame_a_matrix_size);
//	}
//	cufftHandle plan_1d_complex_polar_row, plan_1d_complex_polar_col;
//	{
//		int n_row[] = {frame_polar_matrix_size_rho};
//		int n_col[] = {frame_polar_matrix_size_theta};
//		int inembed_row[] = {frame_polar_matrix_size_rho};
//		int onembed_row[] = {frame_polar_matrix_size_rho};
//		int inembed_col[] = {1};
//		int onembed_col[] = {1};
//		cufftPlanMany(&plan_1d_complex_polar_row, 1, n_row, inembed_row, 1, frame_polar_matrix_size_rho, onembed_row, 1, frame_polar_matrix_size_rho, CUFFT_C2C, frame_polar_matrix_size_theta);
//		cufftPlanMany(&plan_1d_complex_polar_col, 1, n_col, inembed_col, frame_polar_matrix_size_rho, 1, onembed_col, frame_polar_matrix_size_rho, 1, CUFFT_C2C, frame_polar_matrix_size_rho);
//	}
//
//	// Prepare grid, block and shared memory configuration for block matrix extraction
//	const dim3 k0_grid(ceil(frame_a_stream_size / (float) stream_threads_per_block));
//	const dim3 k0_block(stream_threads_per_block);
//
//	// Prepare grid and block configuration for polar transformations
//	const dim3 k1_grid(ceil(frame_stream_size_polar / (float) stream_threads_per_block));
//	const dim3 k1_block(stream_threads_per_block);
//
//	// Transfer input data to device memory
//	cudaMemcpy(frame_a_data, iframe_a_data, frame_a_stream_size_real, cudaMemcpyHostToDevice);
//	cudaMemcpy(frame_b_data, iframe_b_data, frame_b_stream_size_real, cudaMemcpyHostToDevice);
//
//	// Expand and pad frame a
//	Real2ComplexPadAndShift<false, false>
//	<<<k0_grid, k0_block>>>
//	(
//		frame_a_data, frame_a_data_complex,
//		frame_a_stream_size,
//		frame_a_matrix_size, frame_a_matrix_size,
//		NULL
//	);
//
//	// Expand and pad frame b
//	Real2ComplexPadAndShift<false, false>
//	<<<k0_grid, k0_block>>>
//	(
//		frame_b_data, frame_b_data_complex,
//		frame_a_stream_size,
//		frame_b_matrix_size, frame_a_matrix_size,
//		NULL
//	);
//
//	// FFT frame a first row-wise then column-wise
//	cufftExecC2C(plan_1d_complex_row, frame_a_data_complex, frame_a_data_complex_copy, CUFFT_FORWARD);
//	cufftExecC2C(plan_1d_complex_col, frame_a_data_complex_copy, frame_a_data_complex_copy, CUFFT_FORWARD);
//
//	if (rotation_allowed)
//	{
//		// Generate gaussian high pass filter kernel
//		HighPassKernel<true>
//		<<<k0_grid, k0_block>>>
//		(
//			highpass_kernel_complex,
//			frame_a_stream_size,
//			frame_a_matrix_size,
//			0.3f
//		);
//
//		// FFT gaussian high pass kernel and frame b first row-wise then column-wise
//		cufftExecC2C(plan_1d_complex_row, frame_b_data_complex, frame_b_data_complex_copy, CUFFT_FORWARD);
//		cufftExecC2C(plan_1d_complex_col, frame_b_data_complex_copy, frame_b_data_complex_copy, CUFFT_FORWARD);
//		cufftExecC2C(plan_1d_complex_row, highpass_kernel_complex, highpass_kernel_complex, CUFFT_FORWARD);
//		cufftExecC2C(plan_1d_complex_col, highpass_kernel_complex, highpass_kernel_complex, CUFFT_FORWARD);
//
//		// High pass filter both frame's magnitudes
//		ComplexPointwiseMagnitudeMulAndScale<true>
//		<<<k0_grid, k0_block>>>
//		(
//			frame_a_data_complex_copy, highpass_kernel_complex, frame_a_data_complex_filtered,
//			frame_a_stream_size,
//			frame_a_matrix_size,
//			(frame_a_matrix_size * frame_a_matrix_size)
//		);
//
//		ComplexPointwiseMagnitudeMulAndScale<true>
//		<<<k0_grid, k0_block>>>
//		(
//			frame_b_data_complex_copy, highpass_kernel_complex, frame_b_data_complex_filtered,
//			frame_a_stream_size,
//			frame_a_matrix_size,
//			(frame_a_matrix_size * frame_a_matrix_size)
//		);
//
//		// Transform both frames FFT coefficients to polar space
//		Cartesian2PolarTransform
//		<<<k1_grid, k1_block>>>
//		(
//			frame_a_data_complex_filtered, frame_a_data_complex_polar,
//			frame_stream_size_polar,
//			frame_a_matrix_size,
//			frame_polar_matrix_size_rho,
//			frame_polar_matrix_size_theta
//		);
//
//		Cartesian2PolarTransform
//		<<<k1_grid, k1_block>>>
//		(
//			frame_b_data_complex_filtered, frame_b_data_complex_polar,
//			frame_stream_size_polar,
//			frame_a_matrix_size,
//			frame_polar_matrix_size_rho,
//			frame_polar_matrix_size_theta
//		);
//
//		// FFT filtered magnitude polar frames first row-wise then column-wise
//		cufftExecC2C(plan_1d_complex_polar_row, frame_a_data_complex_polar, frame_a_data_complex_polar, CUFFT_FORWARD);
//		cufftExecC2C(plan_1d_complex_polar_col, frame_a_data_complex_polar, frame_a_data_complex_polar, CUFFT_FORWARD);
//		cufftExecC2C(plan_1d_complex_polar_row, frame_b_data_complex_polar, frame_b_data_complex_polar, CUFFT_FORWARD);
//		cufftExecC2C(plan_1d_complex_polar_col, frame_b_data_complex_polar, frame_b_data_complex_polar, CUFFT_FORWARD);
//
//		// Normalized cross correlation of polar frames
//		ComplexPointwiseNormalizedCorrelation
//		<<<k1_grid, k1_block>>>
//		(
//			frame_a_data_complex_polar, frame_b_data_complex_polar,
//			frame_stream_size_polar,
//			(frame_polar_matrix_size_rho * frame_polar_matrix_size_theta)
//		);
//
//		// Inverse FFT cross correlated polar frame map
//		cufftExecC2C(plan_1d_complex_polar_row, frame_a_data_complex_polar, frame_a_data_complex_polar, CUFFT_INVERSE);
//		cufftExecC2C(plan_1d_complex_polar_col, frame_a_data_complex_polar, frame_a_data_complex_polar, CUFFT_INVERSE);
//
//		// Prepare block-wise maximum argument reduction
//		const dim3 k2_grid(1);
//		const dim3 k2_block(frame_stream_size_polar);
//
//		// Calculate maximum argument index
//		ArgumentMaximumReduction<false>
//		<<<k2_grid, k2_block, frame_stream_size_polar * (sizeof(Complex) + sizeof(int))>>>
//		(
//			frame_a_data_complex_polar,
//			max_indices,
//			NULL
//		);
//
//		int* max_indices_host = new int[1];
//		cudaMemcpy(max_indices_host, max_indices, 1 * sizeof(int), cudaMemcpyDeviceToHost);
//		float base_rotation = M_PI * ((float) max_indices_host[0] / frame_polar_matrix_size_rho) / frame_polar_matrix_size_theta;
//		delete max_indices_host;
//
//		Rotate
//		<<<k0_grid, k0_block>>>
//		(
//			frame_b_data_complex, frame_a_data_complex_copy,
//			frame_a_stream_size,
//			frame_a_matrix_size,
//			base_rotation
//		);
//
//		Rotate
//		<<<k0_grid, k0_block>>>
//		(
//			frame_b_data_complex, frame_b_data_complex_copy,
//			frame_a_stream_size,
//			frame_a_matrix_size,
//			base_rotation + M_PI
//		);
//
//		ComplexPointwiseMeanSquaredError
//		<<<k0_grid, k0_block>>>
//		(
//			frame_a_data_complex,
//			frame_a_data_complex_copy,
//			frame_a_data_complex_filtered,
//			frame_a_stream_size
//		);
//
//		ComplexPointwiseMeanSquaredError
//		<<<k0_grid, k0_block>>>
//		(
//			frame_a_data_complex,
//			frame_b_data_complex_copy,
//			frame_b_data_complex_filtered,
//			frame_a_stream_size
//		);
//
//		// Prepare block-wise maximum argument reduction
//		const dim3 k3_grid(1);
//		const dim3 k3_block(frame_a_stream_size);
//
//		SumReduction
//		<<<k3_grid, k3_block, frame_a_stream_size * (sizeof(Complex) + sizeof(int))>>>
//		(
//			frame_a_data_complex_filtered,
//			&squared_mean_error[0]
//		);
//
//		SumReduction
//		<<<k3_grid, k3_block, frame_a_stream_size * (sizeof(Complex) + sizeof(int))>>>
//		(
//			frame_b_data_complex_filtered,
//			&squared_mean_error[1]
//		);
//
//		Complex* squared_mean_error_a = new Complex[1];
//		Complex* squared_mean_error_b = new Complex[1];
//		cudaMemcpy(squared_mean_error_a, &squared_mean_error[0], 1 * sizeof(Complex), cudaMemcpyDeviceToHost);
//		cudaMemcpy(squared_mean_error_b, &squared_mean_error[1], 1 * sizeof(Complex), cudaMemcpyDeviceToHost);
//
//		frame_b_data_complex = (squared_mean_error_a[0].x < squared_mean_error_b[0].x) ? frame_a_data_complex_copy : frame_b_data_complex_copy;
//
//		delete squared_mean_error_a;
//		delete squared_mean_error_b;
//	}
//
//	// FFT frame b first row-wise then column-wise
//	cufftExecC2C(plan_1d_complex_row, frame_b_data_complex, frame_b_data_complex_copy, CUFFT_FORWARD);
//	cufftExecC2C(plan_1d_complex_col, frame_b_data_complex_copy, frame_b_data_complex_copy, CUFFT_FORWARD);
//
//	PrintDeviceComplexMatrix(frame_a_data_complex_copy, frame_a_matrix_size, frame_a_matrix_size);
//	PrintDeviceComplexMatrix(frame_b_data_complex_copy, frame_a_matrix_size, frame_a_matrix_size);
//
//	/*
//
//
//	// Normalized cross correlation of (optionally rotation corrected) input frames
//	ComplexPointwiseNormalizedCorrelation
//	<<<k0_grid, k0_block>>>
//	(
//		frame_a_data_complex_copy, frame_b_data_complex_copy,
//		frame_a_stream_size,
//		(frame_a_stream_size * frame_a_stream_size)
//	);
//
//	// Inverse FFT cross correlated polar frame map
//	cufftExecC2C(plan_1d_complex_row, frame_a_data_complex_copy, frame_a_data_complex_copy, CUFFT_INVERSE);
//	cufftExecC2C(plan_1d_complex_row, frame_a_data_complex_copy, frame_a_data_complex_copy, CUFFT_INVERSE);
//
//
//	PrintDeviceComplexMatrix(frame_a_data_complex_copy, frame_a_matrix_size, frame_a_matrix_size);
//	*/
//
//	/*
//	Complex* result_host = new Complex[frame_stream_size_polar];
//	cudaMemcpy(result_host, frame_a_data_complex_polar, frame_stream_size_polar * sizeof(Complex), cudaMemcpyDeviceToHost);
//
//	float max_value = 0;
//	unsigned int theta_max = 0;
//
//	for (unsigned int i = 0; i < frame_stream_size_polar; ++i)
//	{
//		if (result_host[i].x > max_value)
//		{
//			unsigned int theta = i / frame_polar_matrix_size_rho;
//			unsigned int rho = i - theta * frame_polar_matrix_size_rho;
//
//			if (rho == 0)
//			{
//				max_value = result_host[i].x;
//				theta_max = theta;
//			}
//		}
//	}
//
//	delete result_host;
//
//	*/
//
//	//PrintMax<<<1,1>>>(max_indices);
//
//	//printf("Rotation: %4.2f°\nQuality:  %4.2f% \n\n", ((theta_max * 180.0f / frame_polar_matrix_size_theta)), (max_value * 100.0f));
//
//	//PrintDeviceComplexMatrix(frame_a_data_complex_polar, frame_polar_matrix_size_rho, frame_polar_matrix_size_theta);
//
//
//
//	// Cleanup
//	/*
//	cufftDestroy(plan_1d_complex_row);
//	cudaFree(frame_data);
//	cudaFree(frame_data_complex);
//	*/
//}


__host__ void SubAspectRegistration(const float* iframe_a_data, const float* iframe_b_data, float* result_frame, const unsigned int frame_a_matrix_size, const unsigned int frame_b_matrix_size, bool weighting_window, bool rotation_allowed)
{
	RotationEstimation<true>(iframe_a_data, iframe_b_data, frame_a_matrix_size, frame_b_matrix_size);
}



float SubAspectRegistrationTest()
{
	const unsigned int matrix_size = 31;
	const unsigned int search_matrix_size = 3;

	// TODO: Highpass is active!!!!!!!!!!!!

	Real A[matrix_size * matrix_size] =
	{
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,3,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	};

	Real B[search_matrix_size * search_matrix_size] =
	{
		3,1,1,
		3,1,1,
		3,1,1,
	};


	float milliseconds = 0;

	Real* C = new Real[matrix_size * matrix_size];

	for (unsigned int i = 0; i < 100; ++i)
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		SubAspectRegistration(A, B, C, matrix_size, search_matrix_size, false, true);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		printf("Elapsed Time: %4.2f ms\n", milliseconds);
	}

	bool print = false;

	if(print)
	{
		PrintHostMatrix(C, matrix_size, matrix_size);
	}

	delete C;

	return milliseconds;
}



}
}
}
}
}

