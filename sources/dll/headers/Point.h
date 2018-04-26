#pragma once

#include <string>

#include "cuda_runtime.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class Point
{
public:
	CUDA_CALLABLE_MEMBER Point(); 
	CUDA_CALLABLE_MEMBER Point(float x, float y, float z);
	CUDA_CALLABLE_MEMBER ~Point();

	float X, Y, Z;
	
	CUDA_CALLABLE_MEMBER Point &operator-(const Point &otherPoint)const;
	CUDA_CALLABLE_MEMBER Point &operator+(const Point &otherPoint)const;
	CUDA_CALLABLE_MEMBER bool operator==(const Point &otherPoint)const;
	CUDA_CALLABLE_MEMBER Point &operator=(const Point &otherPoint);
	CUDA_CALLABLE_MEMBER Point &operator+=(const Point &otherPoint);

	CUDA_CALLABLE_MEMBER void MultiplyByLambda(float l);
	CUDA_CALLABLE_MEMBER void DevideByLambda(float l);
	CUDA_CALLABLE_MEMBER void Normalize();
	CUDA_CALLABLE_MEMBER float Length();
	std::string ToFile();
	
	CUDA_CALLABLE_MEMBER static Point GetMidlePoint(const Point &p1, const Point &p2);
	CUDA_CALLABLE_MEMBER static float DotProduct(const Point &p1, const Point &p2);
	CUDA_CALLABLE_MEMBER static Point CrossProduct(const Point &p1, const Point &p2);
	CUDA_CALLABLE_MEMBER static float Distance(const Point &p1, const Point &p2);

	CUDA_CALLABLE_MEMBER static bool CompFloat(float f1, float f2, float e);
};