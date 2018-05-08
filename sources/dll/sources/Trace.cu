#include "Trace.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Log.h"

#include "Camera.h"
#include "LightSource.h"
#include "Triangle.h"
#include "Point.h"

Camera *dev_cameras;
LightSource *dev_lights;
Triangle *dev_triangles;
int dev_triangles_len, dev_lights_len, dev_cameras_len;

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

	dev_cameras_len = s->cameras.size();
	dev_triangles_len = s->triangles.size();
	dev_lights_len = s->lights.size();
}

void StartSequential()
{
	WriteLog("Started sequential GPU trace", true, Log::Trace);
	SequentialTrace << <1, 1 >> > (dev_triangles, dev_lights, dev_cameras, dev_triangles_len, dev_lights_len, dev_cameras_len);
	WriteLog("Finished sequential GPU trace", true, Log::Trace);

}

void startParallel(int block, int thread) //cameras / sampling
{
	WriteLog("Started parallel GPU trace", true, Log::Trace);
	ParallelTrace<<<block, thread>>>(dev_triangles, dev_lights, dev_cameras, dev_triangles_len, dev_lights_len, dev_cameras_len);
	WriteLog("Finished parallel GPU trace", true, Log::Trace);

}

__global__ void SequentialTrace(Triangle *dev_triangles, LightSource *dev_lights, Camera *dev_cameras, int dev_triangles_len, int dev_lights_len, int dev_cameras_len)
{
	Triangle::InitCuRand();
	for (int j = 0; j < dev_cameras_len; j++)
	{
		for (int i = 0; i < SAMPLING; i++)
		{
			printf("LookNum: %d \n", i);

			Point ray = Triangle::GetPointOnSphere(dev_cameras[j].origin);
			Vector vector(dev_cameras[j].origin, ray);
			float a = Trace(dev_lights, dev_triangles, &vector, MAX_DEPT, dev_triangles_len, dev_lights_len, dev_cameras_len);
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

__device__ float Trace(LightSource* dev_lights, Triangle *dev_triangles, Vector *startPoint, int dept, int dev_triangles_len, int dev_lights_len, int dev_cameras_len)
{
	for (int i = 0; i < dept; i++)
	{
		LightSource **directHitLights = new LightSource*[dev_lights_len];
		Point rayToPoint;
		int j = 0;
		for(int k = 0; k < dev_lights_len; k++)
		{
			rayToPoint = dev_lights[k].location - startPoint->Location;
			rayToPoint.Normalize();
			if (Camera::LightHitBeforeTriangle(dev_lights[k], dev_triangles, Vector(startPoint->Location, rayToPoint), dev_triangles_len))
			{
				directHitLights[j++] = &dev_lights[k];
			}
		}
		if (j > 0)
		{
			int max = 0;
			int idx = 0;
			for (int k = 0; k < dev_lights_len; k++)
			{
				LightSource *aktLight = directHitLights[0];
				if (directHitLights[k] && aktLight->intensity < directHitLights[k]->intensity )
				{
					max = idx;
				}
				idx++;
			}
			startPoint->Direction = rayToPoint;
			LightSource *aktLight = directHitLights[0];
			return aktLight->intensity;
		}
		std::pair<Triangle*, Point*> *trianglePointPair = Triangle::ClosestTriangleHit(dev_triangles, *startPoint, dev_triangles_len);

		if (trianglePointPair->first && trianglePointPair->second)
		{
			Triangle triangleHit = *trianglePointPair->first;
			Point pointHit = *trianglePointPair->second;
			Point offset(startPoint->Direction);
			offset.MultiplyByLambda(-1);
			offset.MultiplyByLambda(0.001f);
			pointHit = pointHit + offset;

			bool backfacing = Point::DotProduct(triangleHit.normal, startPoint->Direction) > 0;

			startPoint = &Vector(pointHit, Triangle::GetPointOnHalfSphere(triangleHit, backfacing));
		}
	}
	return 0;
}

__global__ void ParallelTrace(Triangle *dev_triangles, LightSource *dev_lights, Camera *dev_cameras, int dev_triangles_len, int dev_lights_len, int dev_cameras_len)
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