#pragma once

#include <vector>

#include <tuple>

#include "Point.h"
#include "Vector.h"

#include "cuda_runtime.h"

#include "curand.h"
#include "curand_kernel.h"

class Triangle
{
public:
	CUDA_CALLABLE_MEMBER Triangle(Point p0, Point p1, Point p2);
	CUDA_CALLABLE_MEMBER ~Triangle();

	Point p0;
	Point p1;
	Point p2;

	Point normal;

	CUDA_CALLABLE_MEMBER bool operator==(const Triangle &otherTriangle) const;

	CUDA_CALLABLE_MEMBER Point *InsideTriangle(Vector ray);

	CUDA_CALLABLE_MEMBER static std::pair<Triangle*, Point*> *ClosestTriangleHit(std::vector<Triangle*> triangles, Vector ray);

	__device__ static Point GetPointOnSphere(const Point &origin);
	__device__ static Point GetPointOnHalfSphere(Triangle triangle, bool backfacing);

	__device__ static void InitCuRand();

private:
	CUDA_CALLABLE_MEMBER void CalcNormal();
	__device__ static float RandomNumber(float min, float max);
};