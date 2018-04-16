#include "Trace.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Camera.h"
#include "LightSource.h"
#include "Triangle.h"

__device__ Camera *d_cameras;
__device__ LightSource *d_lights;
__device__ Triangle *d_triangles;

cudaError CopyToDevice(Scene * s)
{
	cudaError e = cudaMemcpyToSymbol(d_cameras, s->cameras[0], s->cameras.size());
	if (e != cudaError::cudaSuccess)
	{
		return e;
	}
	e = cudaMemcpyToSymbol(d_lights, s->lights[0], s->lights.size());
	if (e != cudaError::cudaSuccess)
	{
		return e;
	}
	e = cudaMemcpyToSymbol(d_triangles, s->triangles[0], s->triangles.size());

	return e;
}

__global__ void SequentialTrace()
{
	
	//d_cameras->lookDirections[1] = d_cameras->lookDirections[0];
}

__global__ void ParallelTrace()
{

}

cudaError CopyFromDevice(Scene * s)
{
	cudaError e = cudaMemcpyFromSymbol(s->cameras[0], d_cameras, s->triangles.size());

	return e;
}

