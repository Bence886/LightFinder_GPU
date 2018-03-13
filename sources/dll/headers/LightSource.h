#pragma once

#include <vector>

#include "Point.h"
#include "Vector.h"

class LightSource
{
public:
	LightSource(Point loc, float intensity);
	~LightSource();

	Point location;
	float intensity;

	bool operator==(const LightSource &otherLight)const;

	bool IntersectLight(Vector ray);
	
	static LightSource &ClosestLightHit(std::vector<LightSource> lights, Vector ray);
};

