#pragma once

#include <vector>

#include "Point.h"
#include "Vector.h"

#include "cuda_runtime.h"

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

