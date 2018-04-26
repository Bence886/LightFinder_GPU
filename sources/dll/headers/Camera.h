#pragma once

#include "thrust\device_vector.h"

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

#define SAMPLING 100

class Camera
{
public:
	CUDA_CALLABLE_MEMBER Camera(const Point &o);
	CUDA_CALLABLE_MEMBER ~Camera();

	int lookNum = SAMPLING;
	Point origin;
	Point lookDirections[SAMPLING];
	int maxDept;

	static cudaError CopyToSymbol(Camera *cam, Camera *d_cam);
	static cudaError CopyFromSymbol(Camera *d_cam, Camera *cam);

	CUDA_CALLABLE_MEMBER bool operator==(const Camera &otherCamera)const;

	CUDA_CALLABLE_MEMBER void StartCPUTrace(std::vector<LightSource*> lights, std::vector<Triangle*> triangles);
private:
	float CpuTrace(const std::vector<LightSource*> &lights,const std::vector<Triangle*> triangles, Vector *ray, int dept);
	CUDA_CALLABLE_MEMBER bool LightHitBeforeTriangle(const LightSource &light, const std::vector<Triangle*> triangles, const Vector &ray);
};

