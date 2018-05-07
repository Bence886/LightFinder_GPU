#include "Vector.h"
#include "Log.h"

Vector::Vector(Point location, Point direction) : Length(1), Location(location)
{
	if (Point::CompFloat(direction.Length(), 1.0f, 0.0001f))
	{
		direction = direction;
	}
	else
	{
		WriteLog("Direction vector is not a Unit vector!",true, Log::Exception);
	}
	Direction = direction;
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