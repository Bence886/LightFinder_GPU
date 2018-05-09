#include "Triangle.h"

#include "Log.h"

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

Point *Triangle::InsideTriangle(Vector ray)
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
		return NULL;
	}

	f = 1 / a;
	s = ray.Location - p0;
	u = f * (Point::DotProduct(s, h));
	if (u < 0.0 || u > 1.0)
	{
		return NULL;
	}

	q = Point::CrossProduct(s, e1);

	v = f * Point::DotProduct(ray.Direction, q);
	if (v < 0.0 || u + v > 1.0)
	{
		return NULL;
	}
	float t = f * Point::DotProduct(e2, q);
	if (t > 0.00001)
	{
		return new Point(
			ray.Location.X + ray.Direction.X * t,
			ray.Location.Y + ray.Direction.Y * t,
			ray.Location.Z + ray.Direction.Z * t);
	}
	return NULL;
}

void Triangle::CalcNormal()
{
	Point u = (p1 - p0);
	Point v = (p2 - p0);

	normal = Point::CrossProduct(u, v);
	normal.Normalize();
}

#ifdef __CUDACC__
__device__ curandState state;
float Triangle::RandomNumber(float Min, float Max)
{
	float x = curand_uniform(&state);
	x = x *2 -1;
	//printf("Generated random number: %f\n", x);
	return x;
}
void Triangle::InitCuRand()
{
	int id = threadIdx.x + blockIdx.x * 64;
	curand_init(1234, id, 0, &state);
	printf("Init Curand id: %d\n", id);
}
#else
float Triangle::RandomNumber(float Min, float Max)
{
	return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
}
#endif

Point Triangle::GetPointOnSphere(const Point & origin)
{
	Point randomPoint(RandomNumber(-1, 1), RandomNumber(-1, 1), RandomNumber(-1, 1));
	while (randomPoint.Length() > 1)
	{
		randomPoint = Point(RandomNumber(-1, 1), RandomNumber(-1, 1), RandomNumber(-1, 1));
	}
	randomPoint.Normalize();
	return randomPoint;
}

Point Triangle::GetPointOnHalfSphere(Triangle hitTriangle, bool backfacing)
{
	Point normal = hitTriangle.normal;
	if (backfacing)
	{
		normal.MultiplyByLambda(-1);
	}
	Point direction = Point::CrossProduct(normal, hitTriangle.p1 - hitTriangle.p0);
	direction.Normalize();
	Point cross = Point::CrossProduct(normal, direction);

	float x, y, z;
	x = RandomNumber(-1, 1);
	y = RandomNumber(-1, 1);
	z = RandomNumber(0, 1);

	Point randomPoint(
		x * direction.X + y * cross.X + z * normal.X,
		x * direction.Y + y * cross.Y + z * normal.Y,
		x * direction.Z + y * cross.Z + z * normal.Z);

	randomPoint.Normalize();
	return randomPoint;
}

Pair Triangle::ClosestTriangleHit(Triangle *triangles, Vector ray, int triangles_len)
{
	Point *closest = NULL;
	Triangle *hitTriangle = NULL;
	for (int i = 0; i < triangles_len; i++)
	{
		Point *hit = triangles[i].InsideTriangle(ray);
		if (!closest || (hit && (Point::Distance(ray.Location, *hit) < Point::Distance(ray.Location, *closest))))
		{
			hitTriangle = &triangles[i];
			closest = hit;
		}
	}
	if (!hitTriangle)
	{
		return Pair();
	}
	return Pair(hitTriangle, closest);
}

