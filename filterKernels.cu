//
// You received this file as part of RRLib
// Robotics Research Library
//
// Copyright (C) AG Robotersysteme TU Kaiserslautern
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
//----------------------------------------------------------------------
/*!\file    projects/aspect_maps_memory/gpu_algorithms/filterKernels.cu
 *
 * \author  Felix Laufer
 *
 * \date    2014-10-10
 *
 * CUDA: Collection of basic complex math operations and kernels
 *
 */
#ifndef __rrlib__aspect_maps_memory__filterKernels_cu__
#define __rrlib__aspect_maps_memory__filterKernels_cu__

#include "projects/aspect_maps_memory/gpu_algorithms/basicComplexMath.cu"

namespace finroc
{
namespace aspect_maps_memory
{
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
}
}

#endif
