#pragma once

#include <vector>

#include "Point.h"
#include "Vector.h"

#include "cuda_runtime.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class LightSource
{
public:
	CUDA_CALLABLE_MEMBER LightSource(Point loc, float intensity);
	CUDA_CALLABLE_MEMBER ~LightSource();

	Point location;
	float intensity;

	CUDA_CALLABLE_MEMBER bool operator==(const LightSource &otherLight)const;

	CUDA_CALLABLE_MEMBER bool IntersectLight(Vector ray);
	
	CUDA_CALLABLE_MEMBER static LightSource *ClosestLightHit(std::vector<LightSource> lights, Vector ray);
};

