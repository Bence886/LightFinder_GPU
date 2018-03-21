#pragma once

#include "Point.h"

#include "cuda_runtime.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class Vector
{
public:
	CUDA_CALLABLE_MEMBER Vector(Point location, Point direction);
	CUDA_CALLABLE_MEMBER ~Vector();

	Point Location;
	Point Direction;
	int Length;

	CUDA_CALLABLE_MEMBER Point GetEndPoint();

	CUDA_CALLABLE_MEMBER bool operator==(const Vector &otherVector)const;
	CUDA_CALLABLE_MEMBER Vector &operator=(const Vector &otherVector);
};

