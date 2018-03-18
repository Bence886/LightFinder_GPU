#pragma once

#include <string>
#include <vector>

#include "tinyxml2.h"

#include "LightSource.h"
#include "Triangle.h"
#include "Camera.h"

class MyXMLReader
{
public:
	MyXMLReader(std::string filename);
	~MyXMLReader();

	tinyxml2::XMLDocument doc;

	std::vector<LightSource*> GetLightSources();
	std::vector<Triangle*> GetTriangles();
	std::vector<Camera*> GetCameras();
};