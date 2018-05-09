#pragma once

#include "Point.h"
class Triangle;

class Pair
{
public:
	CUDA_CALLABLE_MEMBER Pair();
	CUDA_CALLABLE_MEMBER Pair(Triangle * first, Point * second);
	CUDA_CALLABLE_MEMBER ~Pair();
	Triangle *first;
	Point *second;
	bool empty;
};