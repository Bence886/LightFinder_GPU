#include "Trace.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Camera.h"
#include "LightSource.h"
#include "Triangle.h"
#include "Point.h"

Camera *dev_cameras;
LightSource *dev_lights;
Triangle *dev_triangles;

cudaError CopyToDevice(Scene * s)
{
	cudaError e = cudaSuccess;
	e = cudaMalloc((void**)&dev_triangles, sizeof(Triangle) * s->triangles.size());
	if (e != cudaSuccess)
	{
		return e;
	}
	for (int i = 0; i < s->triangles.size(); i++)
	{
		e = cudaMemcpy(&(dev_triangles[i]), s->triangles[i], sizeof(Triangle), cudaMemcpyHostToDevice);
		if (e != cudaSuccess)
		{
			return e;
		}
	}

	e = cudaMalloc((void**)&dev_lights, sizeof(LightSource) * s->lights.size());
	if (e != cudaSuccess)
	{
		return e;
	}
	for (int i = 0; i < s->lights.size(); i++)
	{
		e = cudaMemcpy(&(dev_lights[i]), s->lights[i], sizeof(LightSource), cudaMemcpyHostToDevice);
		if (e != cudaSuccess)
		{
			return e;
		}
	}

	e = cudaMalloc((void**)&dev_cameras, sizeof(Camera) * s->cameras.size());
	if (e != cudaSuccess)
	{
		return e;
	}
	for (size_t i = 0; i < s->cameras.size(); i++)
	{
		e = cudaMemcpy(&(dev_cameras[0]), s->cameras[i], sizeof(Camera), cudaMemcpyHostToDevice);
		if (e != cudaSuccess)
		{
			return e;
		}
	}
	return e;
}

void StartSequential()
{
	SequentialTrace << <1, 1 >> > (dev_triangles, dev_lights, dev_cameras);
}

void startParallel()
{
}

__global__ void SequentialTrace(Triangle *dev_triangles, LightSource *dev_lights, Camera *dev_cameras)
{
	dev_cameras->lookDirections[0] = Point(6, 87, 79);
	printf("%f\n", dev_cameras->lookDirections[0].X);
}

__global__ void ParallelTrace()
{

}

cudaError CopyFromDevice(Scene * s)
{
	cudaError e = cudaSuccess;
	for (int i = 0; i < s->cameras.size(); i++)
	{
		e = cudaMemcpy(s->cameras[i], &(dev_cameras[i]), sizeof(Camera), cudaMemcpyDeviceToHost);
		if (e != cudaSuccess)
		{
			return e;
		}
	}

	return e;
}