#include "Scene.h"

#include "MyXMLReader.h"

Scene::Scene(std::string filename)
{
	ReadInputFile(filename);
}

Scene::~Scene()
{
}

void Scene::StartTrace_CPU()
{
}

void Scene::CreateFloor(float z)
{
	triangles.push_back(Triangle(Point(100, -100, z), Point(100, 100, z), Point(-100, 100, z)));
	triangles.push_back(Triangle(Point(-100, 100, z), Point(-100, -100, z), Point(100, -100, z)));
}

void Scene::ReadInputFile(std::string fliename)
{
}
