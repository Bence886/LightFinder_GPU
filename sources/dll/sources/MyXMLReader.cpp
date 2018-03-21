#include "MyXMLReader.h"

#include "Log.h"

MyXMLReader::MyXMLReader(std::string filename)
{
	doc.LoadFile(filename.c_str());
}


MyXMLReader::~MyXMLReader()
{
}

std::vector<LightSource*> MyXMLReader::GetLightSources()
{
	WriteLog("Parsing lights", true, Log::Debug);
	std::vector<LightSource*> lights;

	tinyxml2::XMLElement *lightsElement = doc.FirstChildElement("root")->FirstChildElement("lights");

	for (tinyxml2::XMLElement* child = lightsElement->FirstChildElement("light"); child; child = child->NextSiblingElement())
	{
		lights.push_back(
			new LightSource(
				Point(atof(child->Attribute("posx")),
					atof(child->Attribute("posy")),
					atof(child->Attribute("posz"))),
				atof(child->Attribute("intensity"))
			)
		);
	}
	WriteLog(std::string("Lights found: ") + std::to_string(lights.size()), true, Log::Debug);
	return lights;
}

std::vector<Triangle*> MyXMLReader::GetTriangles()
{
	WriteLog("Parsing treiangles", true, Log::Debug);
	std::vector<Triangle*> triangles;

	tinyxml2::XMLElement *trianglesElement = doc.FirstChildElement("root")->FirstChildElement("triangles");

	for (tinyxml2::XMLElement* child = trianglesElement->FirstChildElement("triangle"); child; child = child->NextSiblingElement())
	{
		triangles.push_back(
			new Triangle(
				Point(atof(child->FirstChildElement("point0")->Attribute("posx")), atof(child->FirstChildElement("point0")->Attribute("posy")), atof(child->FirstChildElement("point0")->Attribute("posz"))),
				Point(atof(child->FirstChildElement("point1")->Attribute("posx")), atof(child->FirstChildElement("point1")->Attribute("posy")), atof(child->FirstChildElement("point1")->Attribute("posz"))),
				Point(atof(child->FirstChildElement("point2")->Attribute("posx")), atof(child->FirstChildElement("point2")->Attribute("posy")), atof(child->FirstChildElement("point2")->Attribute("posz")))
			)
		);
	}
	WriteLog(std::string("Triangles found: ") + std::to_string(triangles.size()), true, Log::Debug);
	return triangles;
}

std::vector<Camera*> MyXMLReader::GetCameras()
{
	WriteLog("Parsing cameras", true, Log::Debug);
	std::vector<Camera*> cameras;

	tinyxml2::XMLElement *cameraElement = doc.FirstChildElement("root")->FirstChildElement("cameras");

	for (tinyxml2::XMLElement* child = cameraElement->FirstChildElement("camera"); child; child = child->NextSiblingElement())
	{
		cameras.push_back(
			new Camera(
				Point(atof(child->Attribute("posx")), atof(child->Attribute("posy")), atof(child->Attribute("posz")))));
	}
	WriteLog(std::string("Cameras found: ") + std::to_string(cameras.size()), true, Log::Debug);
	return cameras;
}

