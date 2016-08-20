//----------------------------------------------------------------------
/*!\file    gpu_algorithms/filterKernels.cu
 *
 * \author  Felix Laufer
 *
 *
 * CUDA: Collection of filter kernels
 *
 */
//----------------------------------------------------------------------

#include "gpu_algorithms/basicComplexMath.cu"

namespace gpu_algorithms
{
namespace cuda
{

//----------------------------------------------------------------------
// Helper functions
//----------------------------------------------------------------------

static inline __device__ __host__ float GaussianLowPass(const unsigned int x, const unsigned int y, const float sigma)
{
	const float s = 2 * sigma * sigma;
	return 1.0f / (M_PI * s) * (expf(-(x * x + y * y) / s));
}

static inline __device__ __host__ float GaussianHighPass(const unsigned int x, const unsigned int y, const float sigma)
{
	const float gaussian_lowpass = GaussianLowPass(x, y, sigma);
	return (x == 0 && y == 0) ? 2.0f - gaussian_lowpass : -gaussian_lowpass;
}

//----------------------------------------------------------------------
// Kernel functions
//----------------------------------------------------------------------

template<bool param_shift>
static __global__ void GaussianHighPassKernel(Complex *odata, const unsigned int size, const unsigned int nx, const float sigma)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int offset = (nx - 1) / 2;

	for (unsigned int i = threadID; i < size; i += numThreads)
	{
		int y = i / nx;
		int x = i - y * nx;

		float gaussian_highpass = GaussianHighPass(x, y, sigma);

		unsigned int index = i;
		if (param_shift)
		{
			index = SequentialIndex2DFFTShift(x, y, nx);
		}

		odata[index] = (Complex) {gaussian_highpass, 0.0f};
	}
}

static __global__ void RaisedCosineWindow(Real *odata, const unsigned int nx, const unsigned int ny)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (unsigned int i = threadID; i < nx * ny; i += numThreads)
	{
		const unsigned int y = i / nx;
		const unsigned int x = i - y * nx;
		odata[i] = 0.5f * (1.0f - cosf((2.0f * M_PI * x) / (nx - 1))) * 0.5f * (1.0f - cosf((2.0f * M_PI * y) / (ny - 1)));
	}
}

}
}
