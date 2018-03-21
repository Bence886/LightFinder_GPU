#include "Scene.h"

#include <iostream>

#include "MyXMLReader.h"

#include "Log.h"

Scene::Scene(std::string filename)
{
	ReadInputFile(filename);
}

Scene::~Scene()
{
}

void Scene::StartTrace_CPU()
{
	CreateFloor(-1);

	WriteLog("CPU trace started: ", true, Log::Message);
	WriteLog(std::string("Sampling: ") + std::to_string(cameras[0]->sampling), true, Log::Debug);

	for (Camera *item : cameras)
	{
		item->StartCPUTrace(lights, triangles);
	}
}

void Scene::CreateFloor(float z)
{
	triangles.push_back(new Triangle(Point(100, -100, z), Point(100, 100, z), Point(-100, 100, z)));
	triangles.push_back(new Triangle(Point(-100, 100, z), Point(-100, -100, z), Point(100, -100, z)));
	WriteLog(std::string("Created floor at z level: ") + std::to_string(z), true, Log::Debug);
}

void Scene::ReadInputFile(std::string filename)
{
	MyXMLReader xmlDoc(filename);

	if (!xmlDoc.doc.Error())
	{
		lights = xmlDoc.GetLightSources();
		cameras = xmlDoc.GetCameras();
		triangles = xmlDoc.GetTriangles();
	}
	else
	{
		WriteLog(xmlDoc.doc.ErrorStr(), true, Log::Error);
	}
}