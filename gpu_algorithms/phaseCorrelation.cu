//----------------------------------------------------------------------
/*!\file    gpu_algorithms/phaseCorrelation.h
 *
 * \author  Felix Laufer
 *
 *
 * CUDA: Phase correlation methods
 *
 */
//----------------------------------------------------------------------

#include "gpu_algorithms/basicReductions.cu"
#include "gpu_algorithms/debugPrint.cu"

typedef float2 Vec2f;

namespace gpu_algorithms
{
namespace cuda
{
namespace phase_correlation
{

/**
 * Compute the normalized phase correlation map of two given input frames.
 * Params:
 *  frame_a_complex: frame a
 * 	frame_a_complex: frame b
 * 	nx: width of input frames
 * 	ny: height of input frames
 */
template<bool param_inline, bool param_shift>
static __host__ void CorrelationMap(Complex *frame_a_complex, Complex *frame_b_complex, Complex *correlation_map, unsigned int nx, unsigned int ny)
{
	const unsigned int stream_threads_per_block = 256;

	const unsigned int frame_stream_size = nx * ny;

	// Prepare grid and block configuration for normalized cross correlation
	const dim3 grid(ceil(frame_stream_size / (float) stream_threads_per_block));
	const dim3 block(stream_threads_per_block);

	Complex *frame_a_data_complex,
			*frame_b_data_complex,
			*frame_b_complex_shifted;

	if (param_shift)
	{
		cudaMalloc((void**)&frame_b_complex_shifted, frame_stream_size * sizeof(Complex));
		ComplexStreamSequentialIndex2DFFTShift
		<<<grid, block>>>
		(
			frame_b_complex, frame_b_complex_shifted,
			frame_stream_size, nx
		);
	}
	else
	{
		frame_b_complex_shifted = frame_b_complex;
	}

	if (!param_inline)
	{
		cudaMalloc((void**)&frame_a_data_complex, frame_stream_size * sizeof(Complex) * 2);
		frame_b_data_complex = &frame_a_data_complex[frame_stream_size];
	}
	else
	{
		frame_a_data_complex = frame_a_complex;
		frame_b_data_complex = frame_b_complex;
	}

	// Prepare 1D FFT C2C batched plans
	cufftHandle plan_1d_complex_row, plan_1d_complex_col;
	{
		int n_row[] = {nx};
		int n_col[] = {ny};
		int inembed_row[] = {nx};
		int onembed_row[] = {nx};
		int inembed_col[] = {1};
		int onembed_col[] = {1};
		cufftPlanMany(&plan_1d_complex_row, 1, n_row, inembed_row, 1, nx, onembed_row, 1, nx, CUFFT_C2C, ny);
		cufftPlanMany(&plan_1d_complex_col, 1, n_col, inembed_col, nx, 1, onembed_col, nx, 1, CUFFT_C2C, nx);
	}

	// FFT both frames first row-wise then column-wise
	cufftExecC2C(plan_1d_complex_row, frame_a_complex, frame_a_data_complex, CUFFT_FORWARD);
	cufftExecC2C(plan_1d_complex_col, frame_a_data_complex, frame_a_data_complex, CUFFT_FORWARD);
	cufftExecC2C(plan_1d_complex_row, frame_b_complex_shifted, frame_b_data_complex, CUFFT_FORWARD);
	cufftExecC2C(plan_1d_complex_col, frame_b_data_complex, frame_b_data_complex, CUFFT_FORWARD);

	// Normalized cross correlation of both frames
	ComplexPointwiseNormalizedCorrelation
	<<<grid, block>>>
	(
		frame_a_data_complex, frame_b_data_complex,
		frame_stream_size,
		(nx * ny)
	);

	// Inverse FFT cross correlated map
	cufftExecC2C(plan_1d_complex_row, frame_a_data_complex, correlation_map, CUFFT_INVERSE);
	cufftExecC2C(plan_1d_complex_col, correlation_map, correlation_map, CUFFT_INVERSE);

	// Clean up
	cufftDestroy(plan_1d_complex_row);
	cufftDestroy(plan_1d_complex_col);
	if (!param_inline)
	{
		cudaFree(frame_a_data_complex);
	}
	if (param_shift)
	{
		cudaFree(frame_b_complex_shifted);
	}
}

/**
 * Return the maximum index of the computed normalized cross correlation of two given frames.
 * Params:
 *  frame_a_complex: frame a
 * 	frame_a_complex: frame b
 * 	nx: width
 * 	ny: height
 */
template<bool param_inline, bool param_shift>
static __host__ int PeakIndex(Complex *frame_a_complex, Complex *frame_b_complex, unsigned int nx, unsigned int ny)
{
	const unsigned int max_stream_threads_per_block = 256;

	const unsigned int frame_stream_size = nx * ny;
	const unsigned int stream_threads_per_block = min(max_stream_threads_per_block, frame_stream_size);
	const unsigned int stream_blocks = ceil(frame_stream_size / (float) stream_threads_per_block);

	Complex *correlation_map;

	if (!param_inline)
	{
		cudaMalloc((void**)&correlation_map, frame_stream_size * sizeof(Complex));
	}
	else
	{
		correlation_map = frame_a_complex;
	}

	// Phase correlate both frames and store resulting correlation map in frame a
	CorrelationMap<param_inline, param_shift>(frame_a_complex, frame_b_complex, correlation_map, nx, ny);

	// Calculate the maximum's index
	int peak_index;
	Complex maximum;
	ArgMaxReduce(correlation_map, &maximum, &peak_index, frame_stream_size);

	return peak_index;
}

/**
 * Return the translation vector of the computed normalized cross correlation of two given frames.
 * Params:
 *  frame_a_complex: frame a
 * 	frame_a_complex: frame b
 * 	nx: width and high of input frames
 */
template<bool param_inline>
static __host__ Vec2f TranslationVector(Complex *frame_a_complex, Complex *frame_b_complex, unsigned int nx)
{
	const int peak_index_translation = PeakIndex<param_inline, true>(frame_a_complex, frame_b_complex, nx, nx);

	if (peak_index_translation != -1)
	{
		const unsigned int offset = (nx - 1) / 2;
		int peak_y_translation = (peak_index_translation / nx);
		int peak_x_translation = peak_index_translation - peak_y_translation * nx;
		peak_x_translation -= offset;
		peak_y_translation -= offset;
		return (Vec2f) {peak_x_translation, peak_y_translation};
	}
	else
	{
		return (Vec2f) {0.0f,0.0f};
	}
}

}
}
}