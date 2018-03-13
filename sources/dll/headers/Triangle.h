#pragma once

#include <vector>

#include "Point.h"
#include "Vector.h"

class Triangle
{
public:
	Triangle(Point p0, Point p1, Point p2);
	Triangle();
	~Triangle();

	Point p0;
	Point p1;
	Point p2;

	Point *normal;

	Point InsideTriangle(Vector ray);

	static Triangle &ClosestTriangleHit(std::vector<Triangle> triangles, Vector ray);

private:
	void CalcNormal();
};