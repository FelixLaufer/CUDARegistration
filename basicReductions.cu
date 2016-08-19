#include "gpu_algorithms/basicComplexMath.cu"

namespace gpu_algorithms
{
namespace cuda
{

const unsigned int warpSize = 32;

//----------------------------------------------------------------------
// Helper functions
//----------------------------------------------------------------------

static __inline__ __device__ void WarpSumReduce(Complex *val)
{
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		(*val).x += __shfl_down((*val).x, offset);
		(*val).y += __shfl_down((*val).y, offset);
	}
}

static __inline__ __device__ void WarpArgMaxReduce(Complex *val, int *index)
{
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		Complex val_shuffled;
		val_shuffled.x = __shfl_down((*val).x, offset);
		int index_shuffled = __shfl_down(*index, offset);

		if (val_shuffled.x > (*val).x)
		{
			*val = val_shuffled;
			*index = index_shuffled;
		}
	}
}

static __inline__ __device__ void BlockSumReduce(Complex *val)
{
	static __shared__ Complex shared_val[warpSize];

	const int lane = threadIdx.x % warpSize;
	const int wid = threadIdx.x / warpSize;

	WarpSumReduce(val);

	if (lane == 0)
	{
		shared_val[wid] = *val;
	}

	__syncthreads();

	*val = (threadIdx.x < blockDim.x / warpSize) ? shared_val[lane] : (Complex) {0.0f, 0.0f};

	if (wid == 0)
	{
		WarpSumReduce(val);
	}
}

static __inline__ __device__ void BlockArgMaxReduce(Complex *val, int *index)
{
	static __shared__ Complex shared_val[warpSize];
	static __shared__ int shared_index[warpSize];

	const int lane = threadIdx.x % warpSize;
	const int wid = threadIdx.x / warpSize;

	WarpArgMaxReduce(val, index);

	if (lane == 0)
	{
		shared_val[wid] = *val;
		shared_index[wid] = *index;
	}

	__syncthreads();

	*val = (threadIdx.x < blockDim.x / warpSize) ? shared_val[lane] : (Complex) {0.0f, 0.0f};
	*index = (threadIdx.x < blockDim.x / warpSize) ? shared_index[lane] : -1;

	if (wid == 0)
	{
		WarpArgMaxReduce(val, index);
	}
}

//----------------------------------------------------------------------
// Kernel functions
//----------------------------------------------------------------------

static __global__ void SumReduceKernel(const Complex *idata, Complex *odata, unsigned int stream_size)
{
	Complex sum = (Complex) {0.0f, 0.0f};

	const unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	for (unsigned int i = threadId; i < stream_size; i += blockDim.x * gridDim.x)
	{
		sum.x += idata[i].x;
		sum.y += idata[i].y;
	}

	BlockSumReduce(&sum);

	if (threadIdx.x == 0)
	{
		odata[blockIdx.x] = sum;
	}
}

static __global__ void ArgMaxReduceKernel(const Complex *idata, Complex *odata, int *indices, unsigned int stream_size)
{
	Complex result_val = (Complex) {0.0f, 0.0f};
	int result_index = -1;

	const unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	for (unsigned int i = threadId; i < stream_size; i += blockDim.x * gridDim.x)
	{
		if (idata[i].x > result_val.x)
		{
			result_val = idata[i];
			result_index = i;
		}
	}

	BlockArgMaxReduce(&result_val, &result_index);

	if (threadIdx.x == 0)
	{
		odata[blockIdx.x] = result_val;
		indices[blockIdx.x] = result_index;
	}
}

//----------------------------------------------------------------------
// Host functions
//----------------------------------------------------------------------

static __host__ void SumReduce(const Complex *idata, Complex *odata_host, const unsigned int stream_size)
{
	const int threads = 512;
	const int blocks = min((stream_size + threads - 1) / threads, 1024);

	Complex *out_dev;
	cudaMalloc(&out_dev, sizeof(Complex) * 1024);

	SumReduceKernel<<<blocks, threads>>>(idata, out_dev, stream_size);
	SumReduceKernel<<<1, 1024>>>(out_dev, out_dev, blocks);
	cudaMemcpy(odata_host, out_dev, sizeof(Complex), cudaMemcpyDeviceToHost);

	cudaFree(out_dev);
}

static __host__ void ArgMaxReduce(const Complex *idata, Complex *odata_host, int *index, const unsigned int stream_size)
{
	const int threads = 512;
	const int blocks = min((stream_size + threads - 1) / threads, 1024);

	Complex *out_dev;
	int *indices_dev, *index_dev;
	int indices[1024];

	cudaMalloc(&out_dev, sizeof(Complex) * 1024);
	cudaMalloc(&indices_dev, sizeof(int) * 1024);
	cudaMalloc(&index_dev, sizeof(int));

	ArgMaxReduceKernel<<<blocks, threads>>>(idata, out_dev, indices_dev, stream_size);
	cudaMemcpy(&indices, indices_dev, sizeof(int) * 1024, cudaMemcpyDeviceToHost);
	ArgMaxReduceKernel<<<1, 1024>>>(out_dev, out_dev, index_dev, blocks);
	cudaMemcpy(index, index_dev, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(odata_host, out_dev, sizeof(Complex), cudaMemcpyDeviceToHost);

	*index = indices[*index];

	cudaFree(out_dev);
	cudaFree(indices_dev);
	cudaFree(index_dev);
}


static __host__ void ArgMaxReduceBlockwise(const Complex *idata, Complex *odata_host, int *indices_host, const unsigned int blocks, const unsigned int block_size, const unsigned int stream_size)
{
	const unsigned int threads = min(32, 1024);

	Complex *out_dev;
	int *indices_dev;

	cudaMalloc(&out_dev, sizeof(Complex) * blocks);
	cudaMalloc(&indices_dev, sizeof(int) * blocks);

	ArgMaxReduceKernel<<<blocks, threads>>>(idata, out_dev, indices_dev, stream_size);
	cudaMemcpy(indices_host, indices_dev, sizeof(int) * blocks, cudaMemcpyDeviceToHost);
	cudaMemcpy(odata_host, out_dev, sizeof(Complex) * blocks, cudaMemcpyDeviceToHost);

	cudaFree(out_dev);
	cudaFree(indices_dev);
}

}
}
