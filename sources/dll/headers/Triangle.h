#pragma once

#include <vector>

#include <tuple>

#include "Point.h"
#include "Vector.h"

#include "cuda_runtime.h"

#include "curand.h"
#include "curand_kernel.h"

#include "Pair.h"

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

	CUDA_CALLABLE_MEMBER static Pair ClosestTriangleHit(Triangle *triangles, Vector ray, int triengless_len);

#ifdef __CUDACC__
	__device__ static Point GetPointOnSphere(const Point &origin, curandState *state);
	__device__ static Point GetPointOnHalfSphere(Triangle triangle, bool backfacing, curandState *state);
#else
	static Point GetPointOnSphere(const Point &origin);
	static Point GetPointOnHalfSphere(Triangle triangle, bool backfacing);
#endif

	__device__ static void Dev_InitCuRand(curandState *state);

private:
	CUDA_CALLABLE_MEMBER void CalcNormal();
#ifdef __CUDACC__
	__device__ static float RandomNumber(float min, float max, curandState *state);
#else
	__host__ static float RandomNumber(float min, float max);
#endif 
};