#include "Triangle.h"

#include "Exception.h"

Triangle::Triangle(Point p0, Point p1, Point p2)
{
	this->p0 = p0;
	this->p1 = p1;
	this->p2 = p2;
	CalcNormal();
}

Triangle::~Triangle()
{
}

bool Triangle::operator==(const Triangle &otherTriangle) const {
	return this->p0 == otherTriangle.p0 &&
		this->p1 == otherTriangle.p1 &&
		this->p2 == otherTriangle.p2;
}

Point Triangle::InsideTriangle(Vector ray)
{	//http://geomalgorithms.com/a06-_intersect-2.html
	//http://www.lighthouse3d.com/tutorials/maths/ray-triangle-intersection/
	Point e1, e2, h, s, q;
	float a, f, u, v;
	e1 = p1 - p0;
	e2 = p2 - p0;
	h = Point::CrossProduct(ray.Direction, e2);
	a = Point::DotProduct(e1, h);

	if (a > -0.00001 && a < 0.00001)
	{
		throw new Exception("No Hit!");
	}

	f = 1 / a;
	s = ray.Location - p0;
	u = f * (Point::DotProduct(s, h));
	if (u < 0.0 || u > 1.0)
	{
		throw new Exception("No Hit!");
	}

	q = Point::CrossProduct(s, e1);

	v = f * Point::DotProduct(ray.Direction, q);
	if (v < 0.0 || u + v > 1.0)
	{
		throw new Exception("No Hit!");
	}
	float t = f * Point::DotProduct(e2, q);
	if (t > 0.00001)
	{
		return Point(
			ray.Location.X + ray.Direction.X * t,
			ray.Location.Y + ray.Direction.Y * t,
			ray.Location.Z + ray.Direction.Z * t);
	}
	throw new Exception("No Hit!");
}

Triangle &Triangle::ClosestTriangleHit(std::vector<Triangle> triangles, Vector ray)
{
	Point *closest;
	Triangle *hitTriangle = NULL;
	for(Triangle item : triangles)
	{
		try
		{
			Point hit = item.InsideTriangle(ray);
			if (closest || Point::Distance(ray.Location, hit) < Point::Distance(ray.Location, *closest))
			{
				hitTriangle = &item;
				closest = &hit;
			}
		}
		catch (Exception)
		{
		}
	}
	if (hitTriangle)
	{
		throw new Exception("No triangle hit!");
	}
	return *hitTriangle;
}

void Triangle::CalcNormal()
{
	Point u = (p1 - p0);
	Point v = (p2 - p0);

	normal = &Point::CrossProduct(u, v);
	normal->Normalize();
}