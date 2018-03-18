#include "Point.h"

#include <sstream>
#include "math.h"

Point::Point()
{
}

Point::Point(float x, float y, float z) : X(x), Y(y), Z(z)
{
}

Point::~Point()
{
}

Point & Point::operator-(const Point &otherPoint) const
{
	return Point(X - otherPoint.X, Y - otherPoint.Y, Z - otherPoint.Z);
}

Point & Point::operator+(const Point &otherPoint) const
{
	return Point(X + otherPoint.X, Y + otherPoint.Y, Z + otherPoint.Z);
}


bool Point::operator==(const Point &otherPoint) const 
{
	return (X == otherPoint.X && Y == otherPoint.Y && Z == otherPoint.Z);
}

Point & Point::operator=(const Point &otherPoint)
{
	if (this == &otherPoint)
	{
		return *this;
	}
	X = otherPoint.X;
	Y = otherPoint.Y;
	Z = otherPoint.Z;
	return *this;
}

Point & Point::operator+=(const Point & otherPoint)
{
	this->X += otherPoint.X;
	this->Y += otherPoint.Y;
	this->Z += otherPoint.Z;

	return *this;
}

void Point::MultiplyByLambda(float l)
{
	X *= l;
	Y *= l;
	Z *= l;
}

void Point::DevideByLambda(float l)
{
	X /= l;
	Y /= l;
	Z /= l;
}

void Point::Normalize()
{
	float d = sqrt(X * X + Y * Y + Z * Z);
	if (d != 0)
	{
		DevideByLambda(fabs(d));
	}
}

float Point::Length()
{
	return Distance(Point(0, 0, 0), *this);
}

std::string Point::ToFile()
{
	std::stringstream ss;

	ss << "(" << X << ", " << Y << ", " << Z << ")";

	return ss.str();
}

Point Point::GetMidlePoint(const Point & p1, const Point & p2)
{
	Point midle;
	midle = p1 + p2;
	midle.DevideByLambda(2);

	return midle;
}

float Point::DotProduct(const Point & p1, const Point & p2)
{
	//http://www.lighthouse3d.com/tutorials/maths/inner-product/
	return (p1.X * p2.X + p1.Y * p2.Y + p1.Z * p2.Z);
}

Point Point::CrossProduct(const Point & p1, const Point & p2)
{
	//http://www.lighthouse3d.com/tutorials/maths/vector-cross-product/
	Point  p = Point(p1.Y * p2.Z - p1.Z * p2.Y, p1.Z * p2.X - p1.X * p2.Z, p1.X * p2.Y - p1.Y * p2.X);
	return p;
}

float Point::Distance(const Point & p1, const Point & p2)
{
	return sqrt((p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y) + (p1.Z - p2.Z) * (p1.Z - p2.Z));
}

bool Point::CompFloat(float f1, float f2, float e)
{
	return fabs(f1 - f2) < e;
}