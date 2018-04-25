#include "Camera.h"

#include <vector>
#include <algorithm>

#include <tuple>

#include "MyException.h"
#include "Log.h"
#include "..\headers\Camera.h"

Camera::Camera() {}

Camera::Camera(const Point & o)
{
	origin = o;
	maxDept = 5;
}

Camera::~Camera()
{
}

bool Camera::operator==(const Camera & otherCamera) const
{
	return this->origin == otherCamera.origin &&
		this->maxDept == otherCamera.maxDept;
}

void Camera::StartCPUTrace(std::vector<LightSource*> lights, std::vector<Triangle*> triangles)
{
	for (int i = 0; i < SAMPLING; i++)
	{
		Point ray = Triangle::GetPointOnSphere(origin);
		Vector vector(origin, ray);
		WriteLog(std::string("New ray started to: ") + vector.Direction.ToFile() + " from origin, on camera: " + origin.ToFile(), true, Log::Debug);
		float a = CpuTrace(lights, triangles, &vector, maxDept);
		ray.MultiplyByLambda(a);
		if (a != 0)
		{
			lookDirections[lookNum++] = ray;

		}
		WriteLog(std::string("Look direction found: ") + ray.ToFile(), true, Log::Trace);
	}
}

void Camera::StartGPUTrace()
{
}

float Camera::CpuTrace(const std::vector<LightSource*>& lights, const std::vector<Triangle*> triangles, Vector * startPoint, int dept)
{
	for (int i = 0; i < dept; i++)
	{
		std::vector<LightSource*> directHitLights;
		Point rayToPoint;
		for (LightSource *item : lights)
		{
			rayToPoint = item->location - startPoint->Location;
			rayToPoint.Normalize();
			if (LightHitBeforeTriangle(*item, triangles, Vector(startPoint->Location, rayToPoint)))
			{
				directHitLights.push_back(item);
			}
		}
		if (directHitLights.size() > 0)
		{
			int max = 0;
			int idx = 0;
			for (LightSource *item : directHitLights)
			{
				if (directHitLights[max]->intensity < item->intensity)
				{
					max = idx;
				}
				idx++;
			}
			return directHitLights[max]->intensity;
		}
		std::pair<Triangle*, Point*> *trianglePointPair = Triangle::ClosestTriangleHit(triangles, *startPoint);

		if (trianglePointPair->first && trianglePointPair->second)
		{
			Triangle triangleHit = *trianglePointPair->first;
			Point pointHit = *trianglePointPair->second;
			Point offset(startPoint->Direction);
			offset.MultiplyByLambda(-1);
			offset.MultiplyByLambda(0.001f);
			pointHit = pointHit + offset;

			bool backfacing = Point::DotProduct(*triangleHit.normal, startPoint->Direction) > 0;

			startPoint = &Vector(pointHit, Triangle::GetPointOnHalfSphere(triangleHit, backfacing));
		}
		return 0;
	}
	return 0;
}

bool Camera::LightHitBeforeTriangle(const LightSource & light, const std::vector<Triangle*> triangles, const Vector & ray)
{
	std::pair<Triangle*, Point*> *TrianglePointPair = Triangle::ClosestTriangleHit(triangles, ray);

	Point pointHit = *(*TrianglePointPair).second;
	if (Point::Distance(light.location, ray.Location) < Point::Distance(pointHit, ray.Location))
	{
		return true;
	}
	else
	{
		return false;
	}
	return true;
}

