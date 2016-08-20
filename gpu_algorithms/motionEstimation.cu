//----------------------------------------------------------------------
/*!\file    gpu_algorithms/motionEstimation.cu
 *
 * \author  Felix Laufer
 *
 *
 * CUDA: Algorithms and kernels for fast phase correlation block matching motion estimation on 2d matrices
 *
 */
//----------------------------------------------------------------------

#include <cufft.h>
#include "gpu_algorithms/filterKernels.cu"

typedef float2 Vec2f;

namespace gpu_algorithms
{
namespace cuda
{
namespace motion_estimation
{
 

//----------------------------------------------------------------------
// Kernel functions
//----------------------------------------------------------------------

// Argument of maximum reduction
// Requires: blockDim.x = block stream size
template<bool param_result_maximums>
static __global__ void ArgumentMaximumReduction(const Complex *idata, int *result_indices, Real *result_maximums, const unsigned int maximum_iterations)
{
	extern __shared__ float smem_MaximumArgumentReduction[];
	Complex* sdata_cached = (Complex*) smem_MaximumArgumentReduction;
	Complex* sdata = (Complex*) &sdata_cached[blockDim.x];
	int* sindices = (int*) &sdata[blockDim.x];

	const unsigned int tid = threadIdx.x;
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata_cached[tid].x = idata[i].x;
	sdata_cached[tid].y = 0;

	for (unsigned int n = 0; n < maximum_iterations; ++n)
	{
		sdata[tid]= sdata_cached[tid];
		sindices[tid] = tid;

		__syncthreads();

		int floorPow2 = blockDim.x;
		if (floorPow2 & (floorPow2 - 1))
		{
			while (floorPow2 & (floorPow2 - 1))
			{
				floorPow2 &= floorPow2 - 1;
			}
			if (tid >= floorPow2)
			{
				if (sdata[tid - floorPow2].x < sdata[tid].x)
				{
					sdata[tid - floorPow2] = sdata[tid];
					sindices[tid - floorPow2] = sindices[tid];
				}
			}
			__syncthreads();
		}

		for (unsigned int s = floorPow2 >> 1; s > 0; s >>= 1)
		{
			if (tid < s)
			{
				if (sdata[tid].x < sdata[tid + s].x)
				{
					sdata[tid] = sdata[tid + s];
					sindices[tid] = sindices[tid + s];
				}
			}
			__syncthreads();
		}
 
		if (tid == 0)
		{
			const int last_maximum_index = sindices[0];
			sdata_cached[last_maximum_index] = (last_maximum_index > 0) ? sdata_cached[0] : sdata_cached[1];
			const unsigned int i = blockIdx.x * maximum_iterations + n;
			result_indices[i] = (sdata[0].x > 0.0f) ? last_maximum_index : -1;
			if (param_result_maximums)
			{
				result_maximums[i] = sdata[0].x;
			}
		}

		__syncthreads();
	}
}

// Real to Complex block extraction with optional overlapping, optional circular shift and optional weighting
template<bool param_overlapped, bool param_shift, bool param_weighted>
static __global__ void Real2ComplexMatrixBlockExtraction(const Real *idata, Complex *odata, const unsigned int nx, const unsigned int nx_block, const unsigned int nx_search_block, const Real *weights)
{
	extern __shared__ Real smem_Real2ComplexMatrixBlockExtraction[];

	const unsigned int blocks_matrices_size = blockDim.x * blockDim.y;
	const unsigned int blocks_count_x = ceilf((float) nx / nx_block);

	const unsigned int o_i_block_offset = (blockDim.x - nx_block) / 2;

	const unsigned int block_id = blockIdx.y * gridDim.x + blockIdx.x;

	unsigned int idx_x = threadIdx.x;
	unsigned int idx_y = threadIdx.y;

	int o_block_x = idx_x;
	int o_block_y = idx_y;
 
	const unsigned int i_block_row = block_id / blocks_count_x;
	const unsigned int i_block_col = block_id - i_block_row * blocks_count_x;

	const int i_block_x = o_block_x - o_i_block_offset;
	const int i_block_y = o_block_y - o_i_block_offset;

	Real data;
	if(!param_overlapped && !(0 <= i_block_x && i_block_x < nx_block && 0 <= i_block_y && i_block_y < nx_block))
	{
		data = 0.0f;
	}
	else
	{
		const int i_matrix_x = i_block_col * nx_block + i_block_x;
		const int i_matrix_y = i_block_row * nx_block + i_block_y;
		Real weight = param_weighted ? weights[o_block_y * blockDim.x + o_block_x] : 1.0f;
		const bool is_valid_coordinate = (0 <= i_matrix_x && i_matrix_x < nx && 0 <= i_matrix_y && i_matrix_y < nx);
		data = is_valid_coordinate ? idata[i_matrix_y * nx + i_matrix_x] * weight: 0.0f;
	}

	const unsigned int i = idx_y * blockDim.x + idx_x;
	const unsigned int o_offset = block_id * blocks_matrices_size;

	if (param_shift)
	{
		smem_Real2ComplexMatrixBlockExtraction[SequentialIndex2DFFTShift(o_block_x, o_block_y, nx_search_block)] = data;

		__syncthreads();

		odata[o_offset + i].x = smem_Real2ComplexMatrixBlockExtraction[i];
		odata[o_offset + i].y = 0.0f;
	}
	else
	{
		odata[o_offset + i].x = data;
		odata[o_offset + i].y = 0.0f;
	}
}

// Motion indices to aspect matrix
static __global__ void MotionIndices2Matrix(const int *idata, Vec2f *odata, const unsigned int size, const unsigned int matrix_size, const unsigned int block_size, const unsigned int search_block_size, const bool show_motion = false)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	const unsigned int blocks_count = ceilf((float) matrix_size / block_size);

	for (unsigned int i = threadID; i < size; i += numThreads)
	{
		const unsigned int y = i / matrix_size;
		const unsigned int x = i - y * matrix_size;
		const unsigned int block_row = y / block_size;
		const unsigned int block_col = x / block_size;
		const unsigned int block_id = block_row * blocks_count + block_col;
		const int index = idata[block_id];

		int block_x = 0;
		int block_y = 0;

		if (index > -1)
		{
			block_y = index / search_block_size;
			block_x = index - block_y * search_block_size;
			block_x -= (search_block_size - 1) / 2;
			block_y -= (search_block_size - 1) / 2 ;
		}

		odata[i].x = (show_motion) ? -block_x : block_x;
		odata[i].y = (show_motion) ? -block_y : block_y;
	}
}


//----------------------------------------------------------------------
// Algorithms
//----------------------------------------------------------------------

// Requires: block_size <= search_blocksize <= 32 (CUDA max. threads per block = 32 x 32)
__host__ void BlockMotionEstimation(const float* iframe_a_data, const float* iframe_b_data, Vec2f* omotion_vector_matrix, const unsigned int matrix_size, const unsigned int block_size, const unsigned int search_block_size, const bool weighting_window, const bool show_motion = false)
{
	// Return immediately in case of wrong size specifications
	if (block_size > search_block_size|| block_size > 32 || search_block_size > 32)
	{
		return;
	}

	const unsigned int stream_threads_per_block = 256;

	const unsigned int search_block_size_squared = search_block_size * search_block_size;

	// Number of motion estimation blocks
	const unsigned int matrix_blocks = ceil((float) matrix_size / block_size) * ceil((float) matrix_size / block_size);

	// Stream sizes of raw frame data and matrix block extraction data
	const unsigned int frame_stream_size = matrix_size * matrix_size;
	const unsigned int frame_matrix_block_extraction_stream_size = matrix_blocks * search_block_size_squared;

	// Actual byte sizes of raw frame data and matrix block extraction data
	const unsigned int frame_stream_size_real = frame_stream_size * sizeof(Real);
	const unsigned int frame_matrix_block_extraction_stream_size_complex = frame_matrix_block_extraction_stream_size * sizeof(Complex);

	// Allocate all device memory
	Real *frame_data,
		 *frame_a_data,
		 *frame_b_data,
		 *raised_cosine_window;

	Complex *frame_matrix_block_extraction_complex,
			*frame_a_matrix_block_extraction_complex,
			*frame_b_matrix_block_extraction_complex;

	int *max_indices;

	Vec2f *motion_vector_matrix;

	cudaMalloc((void**)&frame_data, frame_stream_size_real * 2);
	cudaMalloc((void**)&frame_matrix_block_extraction_complex, frame_matrix_block_extraction_stream_size_complex * 2);
	cudaMalloc((void**)&raised_cosine_window, search_block_size_squared * sizeof(Real));
	cudaMalloc((void**)&max_indices, matrix_blocks * sizeof(int));
	cudaMalloc((void**)&motion_vector_matrix, frame_stream_size * sizeof(Vec2f));

	frame_a_data = &frame_data[0];
	frame_b_data = &frame_data[frame_stream_size];
	frame_a_matrix_block_extraction_complex = &frame_matrix_block_extraction_complex[0];
	frame_b_matrix_block_extraction_complex = &frame_matrix_block_extraction_complex[frame_matrix_block_extraction_stream_size];

	// Transfer input data to device memory
	cudaMemcpy(frame_a_data, iframe_a_data, frame_stream_size_real, cudaMemcpyHostToDevice);
	cudaMemcpy(frame_b_data, iframe_b_data, frame_stream_size_real, cudaMemcpyHostToDevice);

	// Prepare matrix block-wise 2D FFT plan
	cufftHandle plan_2d_complex;
	{
		int rank = 2;
		int n[] = {search_block_size, search_block_size};
		int inembed[] = {0, search_block_size};
		int istride = 1;
		int idist = search_block_size_squared;
		int onembed[] = {0, search_block_size};
		int ostride = 1;
		int odist = search_block_size_squared;
		int batch = matrix_blocks;
		cufftPlanMany(&plan_2d_complex, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
	}


	// Prepare grid, block and shared memory configuration for block matrix extraction
	const dim3 k1_grid(matrix_blocks);
	const dim3 k1_block(search_block_size, search_block_size);
	const unsigned int k1_shared_mem_size = search_block_size_squared * sizeof(Real);

	if (weighting_window)
	{
		// Prepare grid and block configuration for raised cosine window
		const dim3 k0_grid(ceil(search_block_size_squared / (float) stream_threads_per_block));
		const dim3 k0_block(stream_threads_per_block);
		RaisedCosineWindow
		<<<k0_grid, k0_block>>>
		(
			raised_cosine_window,
			search_block_size, search_block_size
		);

		// Extract first framed into matrix blocks: overlap, no shift, weighting
		Real2ComplexMatrixBlockExtraction<true, false, true>
		<<<k1_grid, k1_block, k1_shared_mem_size>>>
		(
			frame_a_data, frame_a_matrix_block_extraction_complex,
			matrix_size, block_size, search_block_size,
			raised_cosine_window
		);

		// Extract second frame into matrix blocks: overlap, shift, weighting
		Real2ComplexMatrixBlockExtraction<true, true, true>
		<<<k1_grid, k1_block, k1_shared_mem_size>>>
		(
			frame_b_data, frame_b_matrix_block_extraction_complex,
			matrix_size, block_size, search_block_size,
			raised_cosine_window
		);
	}
	else
	{
		// Extract first framed into matrix blocks: overlap, no shift, no weighting
		Real2ComplexMatrixBlockExtraction<true, false, false>
		<<<k1_grid, k1_block, k1_shared_mem_size>>>
		(
			frame_a_data, frame_a_matrix_block_extraction_complex,
			matrix_size, block_size, search_block_size,
			NULL
		);

		// Extract second frame into matrix blocks: overlap, shift, no weighting
		Real2ComplexMatrixBlockExtraction<true, true, false>
		<<<k1_grid, k1_block, k1_shared_mem_size>>>
		(
			frame_b_data, frame_b_matrix_block_extraction_complex,
			matrix_size, block_size, search_block_size,
			NULL
		);
	}

	// 2D FFT transformation of both frames' matrix blocks
	cufftExecC2C(plan_2d_complex, frame_a_matrix_block_extraction_complex, frame_a_matrix_block_extraction_complex, CUFFT_FORWARD);
	cufftExecC2C(plan_2d_complex, frame_b_matrix_block_extraction_complex, frame_b_matrix_block_extraction_complex, CUFFT_FORWARD);

	// Cross correlate the frames' block matrices
	const dim3 k2_grid(ceil(frame_matrix_block_extraction_stream_size / (float) stream_threads_per_block));
	const dim3 k2_block(stream_threads_per_block);
	ComplexPointwiseNormalizedCorrelation
	<<<k2_grid, k2_block>>>
	(
		frame_a_matrix_block_extraction_complex, frame_b_matrix_block_extraction_complex,
		frame_matrix_block_extraction_stream_size,
		search_block_size_squared
	);

	// 2D FFT transformation of resulting correlation map matrix blocks
	cufftExecC2C(plan_2d_complex, frame_a_matrix_block_extraction_complex, frame_a_matrix_block_extraction_complex, CUFFT_INVERSE);

	// Prepare block-wise maximum argument reduction
	const dim3 k3_grid(matrix_blocks);
	const dim3 k3_block(search_block_size_squared);

	// Calculate block-wise maximum argument indices
	ArgumentMaximumReduction<false>
	<<<k3_grid, k3_block, search_block_size_squared * (2 * sizeof(Complex) + sizeof(int))>>>
	(
		frame_a_matrix_block_extraction_complex,
		max_indices,
		NULL,
		1
	);

	// Calculate motion vectors from motion indices
	const dim3 k4_grid(ceil(frame_stream_size / (float) stream_threads_per_block));
	const dim3 k4_block(stream_threads_per_block);
	MotionIndices2Matrix
	<<<k4_grid, k4_block>>>
	(
		max_indices,
		motion_vector_matrix,
		frame_stream_size,
		matrix_size,
		block_size,
		search_block_size,
		show_motion
	);

	// Transfer result back to host memory
	cudaMemcpy(omotion_vector_matrix, motion_vector_matrix, frame_stream_size * sizeof(Vec2f), cudaMemcpyDeviceToHost);

	// Cleanup
	cufftDestroy(plan_2d_complex);
	cudaFree(frame_data);
	cudaFree(frame_matrix_block_extraction_complex);
	cudaFree(raised_cosine_window);
	cudaFree(max_indices);
	cudaFree(motion_vector_matrix);
}


}
}
}
