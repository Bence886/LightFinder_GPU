#pragma once

#include "Triangle.h"
#include "Point.h"

class Pair
{
public:
	Pair();
	Pair(Triangle * first, Point * second);
	~Pair();
	Triangle *first;
	Point *second;
	bool empty;
};