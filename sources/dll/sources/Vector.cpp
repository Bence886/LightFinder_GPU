#include "Vector.h"
#include "Exception.h"

Vector::Vector(Point location, Point direction)
{
	if (Point::CompFloat(direction.Length(), 1.0f, 0.0001f))
	{
		direction = direction;
	}
	else
	{
		throw Exception();
	}
	Location = location;
	Direction = direction;
	Length = 1;
}

Vector::~Vector()
{
}

Point Vector::GetEndPoint()
{
	Point ret = Direction;
	ret.MultiplyByLambda(Length);
	ret = ret + Location;
	return ret;
}

bool Vector::operator==(const Vector & otherVector) const
{
	return this->Location == otherVector.Location && this->Direction == otherVector.Direction;
}

Vector & Vector::operator=(const Vector & otherVector)
{
	if (this == &otherVector)
	{
		return *this;
	}

	Location = otherVector.Location;
	Direction = otherVector.Direction;

	return *this;
}