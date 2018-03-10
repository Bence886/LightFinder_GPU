#pragma once

#include "cuda_runtime.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class Point
{
public:
	Point();
	CUDA_CALLABLE_MEMBER Point(float x, float y, float z);
	CUDA_CALLABLE_MEMBER ~Point();

	float X, Y, Z;
	
	Point &operator-(const Point &otherPoint);
	Point &operator+(const Point &otherPoint);
	Point &operator=(const Point &otherPoint);
	Point &operator==(const Point &otherPoint);

	void MultipyByLambda(float l);
	void DevideByLambda(float l);
	void Normalize();
	float Length();

	static bool CompFloat(float f1, float f2, float e);

	static Point GetMidlePoint(const Point &p1, const Point &p2);
	static float DotProduct(const Point &p1, const Point &p2);
	static float InnerProduct(const Point &p1, const Point &p2);
	static Point CrossProduct(const Point &p1, const Point &p2);
	static float Distance(const Point &p1, const Point &p2);

	static Point GetPointOnSphere(const Point &origin);
};