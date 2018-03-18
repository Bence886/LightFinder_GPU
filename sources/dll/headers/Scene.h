#pragma once

#include <string>
#include <vector>

#include "LightSource.h"
#include "Triangle.h"
#include "Camera.h"

class Scene
{
public:
	Scene(std::string filename);
	~Scene();

	std::vector<LightSource> lights;
	std::vector<Triangle> triangles;
	std::vector<Camera*> cameras;

	void StartTrace_CPU();

private:
	void CreateFloor(float z);
	void ReadInputFile(std::string fliename);
};

