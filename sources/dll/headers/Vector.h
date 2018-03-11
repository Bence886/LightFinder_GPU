#pragma once

#include "Point.h"

class Vector
{
public:
	Vector(Point location, Point direction);
	~Vector();

	Point Location;
	Point Direction;
	int Length;

	Point GetEndPoint();

	bool operator==(const Vector &otherVector)const;
	Vector &operator=(const Vector &otherVector);
};

