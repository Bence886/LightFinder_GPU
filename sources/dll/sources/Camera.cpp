#include "Camera.h"

Camera::Camera(const Point & o)
{
	origin = o;
	maxDept = 5;
	sampling = 10;
}

Camera::~Camera()
{
}

bool Camera::operator==(const Camera & otherCamera) const
{
	return this->origin == otherCamera.origin &&
		this->maxDept == otherCamera.maxDept &&
		this->sampling == otherCamera.sampling;
}

void Camera::StartCPUTrace()
{
}

void Camera::StartGPUTrace()
{
}

float Camera::CpuTrace(const std::vector<LightSource>& lights, const std::vector<Triangle>& triangles, Vector * ray, int dept)
{
	return 0.0f;
}

float Camera::GPUTrace()
{
	return 0.0f;
}

LightSource & Camera::LightHitBeforeTriangle(const LightSource & light, const std::vector<Triangle>* triangles, const Vector & ray)
{
	throw 0;
}
