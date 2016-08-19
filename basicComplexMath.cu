#include <math.h>
#include <cufft.h>

namespace gpu_algorithms
{
namespace cuda
{

typedef cufftComplex Complex;
typedef cufftReal Real;

//----------------------------------------------------------------------
// Helper functions
//----------------------------------------------------------------------

static __host__ __device__ inline unsigned int SequentialIndex2DCyclicShift(const unsigned int x, const unsigned int y, const unsigned int nx, const unsigned int ny, const int shift_x, const int shift_y)
{
	int xx = x;
	int yy = y;
	xx += shift_x;
	yy += shift_y;
	xx = (xx < 0) ? xx + (int) nx : (xx >= (int) nx) ? xx - (int) nx : xx;
	yy = (yy < 0) ? yy + (int) ny : (yy >= (int) ny) ? yy - (int) ny : yy;

	return yy * nx + xx;
}

// Cyclically shift the matrix
static __host__ __device__ inline unsigned int SequentialIndex2DCyclicShift(const unsigned int x, const unsigned int y, const unsigned int matrix_size, const int shift)
{
	return SequentialIndex2DCyclicShift(x, y, matrix_size, matrix_size, shift, shift);
}

// Cyclically shift the matrix s.t. the kernel's center point corresponds to the first sequential array index
static __host__ __device__ inline unsigned int SequentialIndex2DFFTShift(const unsigned int x, const unsigned int y, const unsigned int matrix_size)
{
	int shift = ((int) matrix_size - 1) / 2;
	return SequentialIndex2DCyclicShift(x, y, matrix_size, -shift);
}

// Cyclically shift the matrix s.t. the first sequential array index to the kernel's center point
static __host__ __device__ inline unsigned int SequentialIndex2DInverseFFTShift(const unsigned int x, const unsigned int y, const unsigned int matrix_size)
{
	int shift = ((int) matrix_size - 1) / 2;
	return SequentialIndex2DCyclicShift(x, y, matrix_size, shift);
}

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scalar multiplication
static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex scalar division
static __device__ __host__ inline Complex ComplexScaleDiv(Complex a, float s)
{
    Complex c;
    c.x = a.x / s;
    c.y = a.y / s;
    return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex division
static __device__ __host__ inline Complex ComplexDiv(Complex a, Complex b)
{
    float divisor = b.x * b.x + b.y * b.y;
	Complex c;
    c.x = (a.x * b.x + a.y * b.y) / divisor;
    c.y = (-a.x * b.y + a.y * b.x) / divisor;
    return c;
}

// Complex bilinear interpolation
static __device__ __host__ inline float BilinearInterpolation(float x, float y, const Complex *data, const unsigned int matrix_size)
{
	int x0 = floor(x);
	int x1 = ceil(x);
	int y0 = floor(y);
	int y1 = ceil(y);

	x0 = (x0 < 0) ? 0 : (x0 >= matrix_size) ? matrix_size - 1 : x0;
	x1 = (x1 < 0) ? 0 : (x1 >= matrix_size) ? matrix_size - 1 : x1;
	y0 = (y0 < 0) ? 0 : (y0 >= matrix_size) ? matrix_size - 1 : y0;
	y1 = (y1 < 0) ? 0 : (y1 >= matrix_size) ? matrix_size - 1 : y1;

	const unsigned int index00 = y0 * matrix_size + x0;
	const unsigned int index01 = y1 * matrix_size + x0;
	const unsigned int index10 = y0 * matrix_size + x1;
	const unsigned int index11 = y1 * matrix_size + x1;

	const float v00 = data[index00].x;
	const float v01 = data[index01].x;
	const float v10 = data[index10].x;
	const float v11 = data[index11].x;

	const float m1 = (abs(y0 - y1) > 0.0f) ? (v00 - v01) / (y0 - y1) : 0.0f;
	const float m2 = (abs(y0 - y1) > 0.0f) ? (v10 - v11) / (y0 - y1) : 0.0f;
	const float b1 = v00 - m1 * y0;
	const float b2 = v10 - m2 * y0;

	const float vm1 = y * m1 + b1;
	const float vm2 = y * m2 + b2;

	const float mi = (abs(x0 - x1) > 0.0f) ? (vm1 - vm2) / (x0 - x1) : 0.0f;
	const float bi = vm1 - mi * x0;

	const float vi = x * mi + bi;

	return vi;
}

//----------------------------------------------------------------------
// Kernel functions
//----------------------------------------------------------------------

static __global__ void ComplexStreamSequentialIndex2DFFTShift(const Complex *idata, Complex *odata, const unsigned int stream_size, const unsigned int matrix_size)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (unsigned int i = threadID; i < stream_size; i += numThreads)
	{
		unsigned int index = i;
		int y = i / matrix_size;
		int x = i - y * matrix_size;
		index = SequentialIndex2DFFTShift(x, y, matrix_size);
		odata[index] = idata[i];
	}
}

// Complex point-wise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex* a, const Complex* b, const unsigned int stream_size, const float normalization_factor)
{
    const unsigned int numThreads = blockDim.x * gridDim.x;
    const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = threadID; i < stream_size; i += numThreads)
    {
        Complex product = ComplexMul(a[i], b[i]);
    	a[i] = (Complex) {product.x / normalization_factor, product.y / normalization_factor};
    }
}

// Complex point-wise normalized correlation
static __global__ void ComplexPointwiseNormalizedCorrelation(Complex* a, const Complex* b, const unsigned int stream_size, const float normalization_factor)
{
    const unsigned int numThreads = blockDim.x * gridDim.x;
    const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = threadID; i < stream_size; i += numThreads)
    {
    	Complex product = ComplexMul((Complex) {a[i].x / normalization_factor, a[i].y / normalization_factor}, (Complex) {b[i].x / normalization_factor, -b[i].y / normalization_factor});
    	Real norm = sqrtf(product.x * product.x + product.y * product.y);
    	Complex result = (norm > 0.0f) ? (Complex) {product.x / norm, product.y / norm} : (Complex) {0.0f, 0.0f};
    	a[i] = (Complex) {result.x / normalization_factor, result.y / normalization_factor};
    }
}

// Complex square matrix transposition
template <unsigned int param_tile_dim, unsigned int param_block_rows>
static __global__ void SquareMatrixTranspose(Complex *idata, const Complex *odata, unsigned int matrix_size)
{
	__shared__ Complex tile[param_tile_dim][param_tile_dim + 1];

	const unsigned int blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
	const unsigned int blockIdx_y = blockIdx.x;

	unsigned int xIndex = blockIdx_x * param_tile_dim + threadIdx.x;
	unsigned int yIndex = blockIdx_y * param_tile_dim + threadIdx.y;

	const unsigned int index_in = xIndex + yIndex * matrix_size;

	xIndex = blockIdx_y * param_tile_dim + threadIdx.x;
	yIndex = blockIdx_x * param_tile_dim + threadIdx.y;

	const unsigned int index_out = xIndex + yIndex * matrix_size;

	for (unsigned int i = 0; i < param_tile_dim; i += param_block_rows)
	{
		tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * matrix_size];
	}

	__syncthreads();

	for (unsigned int i = 0; i < param_tile_dim; i += param_block_rows)
	{
		odata[index_out + i * matrix_size] = tile[threadIdx.x][threadIdx.y+i];
	}
}

// Rotate a complex matrix by the given angle
static __global__ void Rotate(const Complex *idata, Complex *odata, const unsigned int stream_size, const unsigned int matrix_size, const float angle)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	const unsigned int offset = (matrix_size - 1) / 2;

	for (unsigned int i = threadID; i < stream_size; i += numThreads)
	{
		const unsigned int y = i / matrix_size;
		const unsigned int x = i - y * matrix_size;

		const int x_o = x - offset;
		const int y_o = y - offset;

		const float x_r = x_o * cos(angle) - y_o * sin(angle);
		const float y_r = x_o * sin(angle) + y_o * cos(angle);

		const float x_src = x_r + offset;
		const float y_src = y_r + offset;

		odata[i].x = BilinearInterpolation(x_src, y_src, idata, matrix_size);
		odata[i].y = 0.0f;
	}
}

// Translate a complex matrix by the given translation vector
static __global__ void Translate(const Complex *idata, Complex *odata, const unsigned int stream_size, const unsigned int matrix_size, const int x_translation, const int y_translation)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (unsigned int i = threadID; i < stream_size; i += numThreads)
	{
		int y = i / matrix_size;
		int x = i - y * matrix_size;

		x -= x_translation;
		y -= y_translation;

		Complex data = (0 <= x && x < matrix_size && 0 <= y && y < matrix_size) ? idata[y * matrix_size + x] : (Complex) {0.0f, 0.0f};
		odata[i] = data;
	}
}

}
}
