#include "Trace.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Log.h"

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
		WriteLog("Malloc dev_triangles: ", true, Log::Exception);
		return e;
	}
	for (int i = 0; i < s->triangles.size(); i++)
	{
		e = cudaMemcpy(&(dev_triangles[i]), s->triangles[i], sizeof(Triangle), cudaMemcpyHostToDevice);
		if (e != cudaSuccess)
		{
			WriteLog("Copy dev_triangles: ", true, Log::Exception);
			return e;
		}
	}

	e = cudaMalloc((void**)&dev_lights, sizeof(LightSource) * s->lights.size());
	if (e != cudaSuccess)
	{
		WriteLog("Malloc dev_lights: ", true, Log::Exception);
		return e;
	}
	for (int i = 0; i < s->lights.size(); i++)
	{
		e = cudaMemcpy(&(dev_lights[i]), s->lights[i], sizeof(LightSource), cudaMemcpyHostToDevice);
		if (e != cudaSuccess)
		{
			WriteLog("Copy dev_triangles: ", true, Log::Exception);
			return e;
		}
	}

	e = cudaMalloc((void**)&dev_cameras, sizeof(Camera) * s->cameras.size());
	if (e != cudaSuccess)
	{
		WriteLog("Malloc dev_cameras: ", true, Log::Exception);
		return e;
	}
	for (size_t i = 0; i < s->cameras.size(); i++)
	{
		e = cudaMemcpy(&(dev_cameras[0]), s->cameras[i], sizeof(Camera), cudaMemcpyHostToDevice);
		if (e != cudaSuccess)
		{
			WriteLog("Copy dev_triangles: ", true, Log::Exception);
			return e;
		}
	}
	return e;
}

void StartSequential()
{
	WriteLog("Started sequential GPU trace", true, Log::Trace);
	SequentialTrace << <1, 1 >> > (dev_triangles, dev_lights, dev_cameras);
	WriteLog("Finished sequential GPU trace", true, Log::Trace);

}

void startParallel(int block, int thread) //cameras / sampling
{
	WriteLog("Started parallel GPU trace", true, Log::Trace);
	ParallelTrace<<<block, thread>>>(dev_triangles, dev_lights, dev_cameras);
	WriteLog("Finished parallel GPU trace", true, Log::Trace);

}

__global__ void SequentialTrace(Triangle *dev_triangles, LightSource *dev_lights, Camera *dev_cameras)
{
	Triangle::InitCuRand();
	for (int j = 0; j < 1; j++)
	{
		for (int i = 0; i < SAMPLING; i++)
		{
			printf("LookNum: %d \n", i);

			Point ray = Triangle::GetPointOnSphere(dev_cameras[j].origin);
			Vector vector(dev_cameras[j].origin, ray);
			float a = 1;//= CpuTrace(dev_lights, dev_triangles, &vector, MAX_DEPT);
			ray = vector.Direction;
			ray.MultiplyByLambda(a);
			if (a != 0)
			{
				dev_cameras[j].lookDirections[dev_cameras[j].lookNum++] = ray;
				printf("%f\n", dev_cameras[j].lookDirections[i].X);
			}
		}
	}
}

__global__ void ParallelTrace(Triangle *dev_triangles, LightSource *dev_lights, Camera *dev_cameras)
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
			WriteLog("Copy back dev_cameras: ", true, Log::Error);
			return e;
		}
	}

	return e;
}