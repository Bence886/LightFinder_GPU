#pragma once

#include <vector>

#include <tuple>

#include "Point.h"
#include "Vector.h"

class Triangle
{
public:
	Triangle(Point p0, Point p1, Point p2);
	~Triangle();

	Point p0;
	Point p1;
	Point p2;

	Point *normal;

	bool operator==(const Triangle &otherTriangle) const;

	Point InsideTriangle(Vector ray);

	static std::pair<Triangle, Point> &ClosestTriangleHit(std::vector<Triangle> triangles, Vector ray);

	static Point GetPointOnSphere(const Point &origin);
	static Point GetPointOnHalfSphere(Triangle triangle, bool backfacing);


private:
	void CalcNormal();
	static float RandomNumber(float min, float max);
};