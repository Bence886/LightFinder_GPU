#include "Trace.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Camera.h"
#include "LightSource.h"
#include "Triangle.h"

Camera **dev_cameras;
Triangle **dev_triangles;
LightSource **dev_lights;

cudaError CopyToDevice(Scene * s)
{
	cudaError e = cudaSuccess;
	dev_triangles = new Triangle*[s->triangles.size()];
	for (int i = 0; i < s->triangles.size(); i++)
	{
		e = cudaMalloc((void**)&dev_triangles[i], sizeof(Triangle));
		if (e != cudaSuccess)
		{
			return e;
		}
		e = cudaMemcpy(dev_triangles[i], s->triangles[i], sizeof(Triangle), cudaMemcpyHostToDevice);
		if (e != cudaSuccess)
		{
			return e;
		}
	}

	return e;
}

void startSequential()
{
	SequentialTrace << <1, 1 >> > (dev_triangles);
}

__global__ void SequentialTrace(Triangle **dev_triangles)
{
	printf("%d", dev_triangles[0]->p0);
}

__global__ void ParallelTrace()
{

}

cudaError CopyFromDevice(Scene * s)
{
	cudaError e = cudaSuccess;

	return e;
}