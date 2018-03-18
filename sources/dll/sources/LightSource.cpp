#include "LightSource.h"

#include "math.h"
#include "MyException.h"

LightSource::LightSource(Point loc, float intensity) : location(loc), intensity(intensity)
{
}

LightSource::~LightSource()
{
}

bool LightSource::operator==(const LightSource &otherLight)const {
	return this->location == otherLight.location && this->intensity == otherLight.intensity;
}

bool LightSource::IntersectLight(Vector ray)
{
	Point op = location - ray.Location;
	float b = Point::DotProduct(op, ray.Direction);
	float disc = b * b - Point::DotProduct(op, op) + intensity * intensity;
	if (disc < 0)
		return false;
	else disc = (float)sqrt(disc);
	return true;
}

LightSource &LightSource::ClosestLightHit(std::vector<LightSource> lights, Vector ray)
{
	LightSource *closest = NULL;
	for(LightSource item : lights)
	{
		if (item.IntersectLight(ray) && (closest || Point::Distance(item.location, ray.Location) < Point::Distance(closest->location, ray.Location)))
		{
			closest = &item;
		}
	}
	if (closest)
	{
		throw MyException("No light hit");
	}
	return *closest;
}