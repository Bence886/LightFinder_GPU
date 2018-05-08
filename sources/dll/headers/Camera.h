#pragma once

#include "thrust\device_vector.h"

#include "Point.h"
#include "LightSource.h"
#include "Vector.h"
#include "Triangle.h"

#include "cuda_runtime.h"

#define SAMPLING 100
#define MAX_DEPT 5

class Camera
{
public:
	CUDA_CALLABLE_MEMBER Camera(const Point &o);
	CUDA_CALLABLE_MEMBER Camera();
	CUDA_CALLABLE_MEMBER ~Camera();

	int lookNum = 0;
	Point origin;
	Point lookDirections[SAMPLING];

	CUDA_CALLABLE_MEMBER bool operator==(const Camera &otherCamera)const;

#ifdef __CUDACC__
	__device__ void StartCPUTrace(std::vector<LightSource*> lights, std::vector<Triangle*> triangles);
#else
	void StartCPUTrace(std::vector<LightSource*> lights, std::vector<Triangle*> triangles);
#endif

	CUDA_CALLABLE_MEMBER static bool LightHitBeforeTriangle(LightSource &light, Triangle *triangles, const Vector &ray, int triangles_len);

private:
#ifdef __CUDACC__
	__device__ float CpuTrace(const std::vector<LightSource*> &lights, const std::vector<Triangle*> triangles, Vector *ray, int dept);
#else
	float CpuTrace(const std::vector<LightSource*> &lights, const std::vector<Triangle*> triangles, Vector *ray, int dept);
#endif


};

