#pragma once

#include <vector>

#include <tuple>

#include "Point.h"
#include "Vector.h"

#include "cuda_runtime.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class Triangle
{
public:
	CUDA_CALLABLE_MEMBER Triangle(Point p0, Point p1, Point p2);
	CUDA_CALLABLE_MEMBER ~Triangle();

	Point p0;
	Point p1;
	Point p2;

	Point *normal;

	CUDA_CALLABLE_MEMBER bool operator==(const Triangle &otherTriangle) const;

	CUDA_CALLABLE_MEMBER Point InsideTriangle(Vector ray);

	CUDA_CALLABLE_MEMBER static std::pair<Triangle, Point> &ClosestTriangleHit(std::vector<Triangle*> triangles, Vector ray);

	CUDA_CALLABLE_MEMBER static Point GetPointOnSphere(const Point &origin);
	CUDA_CALLABLE_MEMBER static Point GetPointOnHalfSphere(Triangle triangle, bool backfacing);


private:
	CUDA_CALLABLE_MEMBER void CalcNormal();
	CUDA_CALLABLE_MEMBER static float RandomNumber(float min, float max);
};