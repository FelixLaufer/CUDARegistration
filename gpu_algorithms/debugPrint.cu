//----------------------------------------------------------------------
/*!\file    gpu_algorithmsn/BasicComplexMath.cu
 *
 * \author  Felix Laufer
 *
 *
 * CUDA: Output methods for debugging
 *
 */
//----------------------------------------------------------------------

#include <math.h>
#include <cufft.h>
#include <stdio.h>

namespace gpu_algorithms
{
namespace cuda
{

typedef cufftComplex Complex;
typedef cufftReal Real;

// Convert real data stream to complex data stream
static __global__ void Real2Complex(const Real *idata, Complex *odata, const unsigned int stream_size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int i = threadID; i < stream_size; i += numThreads)
  {
    odata[i].x = idata[i];
    odata[i].y = 0.0f;
  }
}

// Convert complex data stream to real data stream
static __global__ void Complex2Real(const Complex *idata, Real *odata, const unsigned int stream_size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int i = threadID; i < stream_size; i += numThreads)
  {
    odata[i] = idata[i].x;
  }
}

// Print a real data stream located in host memory in matrix form
static __host__ void PrintHostMatrix(const Real* idata, const unsigned int nx, const unsigned ny)
{
  for(unsigned int i = 0; i < nx * ny; ++i)
  {
    printf("%10.8lf ", idata[i]);

    if ((i+1) % nx == 0)
    {
      printf("%s\n", "");
    }
  }

  printf("%s\n", "-----------------------------------------------");
}

// Print a complex data stream located in device memory in matrix form
static __host__ void PrintDeviceComplexMatrix(const Complex* idata, const unsigned int nx, const unsigned int ny)
{
  unsigned int stream_size = nx * ny;
  unsigned int stream_size_real = stream_size * sizeof(Real);

  Real* result;
  cudaMalloc((void**)&result, stream_size_real);

  const dim3 grid(ceil(stream_size / 256.0f));
  const dim3 block(256.0f);
  Complex2Real<<<grid, block>>>(idata, result, stream_size);

  Real* result_host = new Real[stream_size];
  cudaMemcpy(result_host, result, stream_size_real, cudaMemcpyDeviceToHost);
  PrintHostMatrix(result_host, nx, ny);

  cudaFree(result);
  delete result_host;
}

// Print a complex real stream located in device memory in matrix form
static __host__ void PrintDeviceRealMatrix(const Real* idata, const unsigned int nx, const unsigned int ny)
{
  unsigned int stream_size = nx * ny;
  unsigned int stream_size_real = stream_size * sizeof(Real);

  Real* result_host = new Real[stream_size];
  cudaMemcpy(result_host, idata, stream_size_real, cudaMemcpyDeviceToHost);
  PrintHostMatrix(result_host, nx, ny);

  delete result_host;
}

}
}