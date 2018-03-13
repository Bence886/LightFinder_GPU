#pragma once

#include "vector"

#include "Point.h"
#include "LightSource.h"
#include "Vector.h"
#include "Triangle.h"

class Camera
{
public:
	Camera(const Point &o);
	~Camera();

	int sampling;
	Point origin;
	std::vector<Point> lookDirections;
	int maxDept;

	bool operator==(const Camera &otherCamera)const;

	void StartTrace();
private:
	float Trace(const std::vector<LightSource> &lights,const std::vector<Triangle> &triangles, Vector *ray, int dept);
	LightSource &LightHitBeforeTriangle(const LightSource &light, const std::vector<Triangle> *triangles, const Vector &ray);
};

