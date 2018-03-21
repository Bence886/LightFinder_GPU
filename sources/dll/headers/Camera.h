#pragma once

#include "vector"

#include "Point.h"
#include "LightSource.h"
#include "Vector.h"
#include "Triangle.h"

#include "cuda_runtime.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class Camera
{
public:
	CUDA_CALLABLE_MEMBER Camera(const Point &o);
	CUDA_CALLABLE_MEMBER ~Camera();

	int sampling;
	Point origin;
	std::vector<Point> lookDirections;
	int maxDept;

	CUDA_CALLABLE_MEMBER bool operator==(const Camera &otherCamera)const;

	CUDA_CALLABLE_MEMBER void StartCPUTrace(std::vector<LightSource*> lights, std::vector<Triangle*> triangles);
	CUDA_CALLABLE_MEMBER void StartGPUTrace();
private:
	float CpuTrace(const std::vector<LightSource*> &lights,const std::vector<Triangle*> triangles, Vector *ray, int dept);
	CUDA_CALLABLE_MEMBER float GPUTrace();
	CUDA_CALLABLE_MEMBER bool LightHitBeforeTriangle(const LightSource &light, const std::vector<Triangle*> triangles, const Vector &ray);
};

