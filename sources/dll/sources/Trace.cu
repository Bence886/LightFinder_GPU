#include "Trace.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Camera.h"
#include "LightSource.h"
#include "Triangle.h"

Camera **cameras;
LightSource **lights;
Triangle **triangles;

cudaError CopyToDevice(Scene * s)
{
	cudaError e;
	cameras = new Camera*[s->cameras.size()];
	int i = 0;
	for (Camera *item : s->cameras)
	{
		e = Camera::CopyToSymbol(item, cameras[i]);
		if (e != cudaSuccess)
		{
			return e;
		}
		i++;
	}
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
	return cudaSuccess;
}