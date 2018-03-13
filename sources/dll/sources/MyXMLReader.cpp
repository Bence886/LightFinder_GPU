#include "MyXMLReader.h"

MyXMLReader::MyXMLReader(std::string filename)
{
	doc.LoadFile(filename.c_str());
}


MyXMLReader::~MyXMLReader()
{
}

std::vector<LightSource> MyXMLReader::GetLightSources()
{
	{
		std::vector<LightSource> lights;

		tinyxml2::XMLElement *lightsElement = doc.FirstChildElement("root")->FirstChildElement("lights");

		for (tinyxml2::XMLElement* child = lightsElement->FirstChildElement(); child; child = child->NextSiblingElement())
		{
			lights.push_back(
				LightSource(
					Point(atof(child->Attribute("posx")),
						atof(child->Attribute("posy")),
						atof(child->Attribute("posz"))),
					atof(child->Attribute("intensity"))
				)
			);
		}

		return lights;
	}
}

std::vector<Triangle> MyXMLReader::GetTriangles()
{
	std::vector<Triangle> triangles;

	tinyxml2::XMLElement *trianglesElement = doc.FirstChildElement("root")->FirstChildElement("triangles");

	for (tinyxml2::XMLElement* child = trianglesElement->FirstChildElement(); child; child = child->NextSiblingElement())
	{
		triangles.push_back(
			Triangle(
				Point(atof(child->FirstChildElement("point0")->Attribute("posx")), atof(child->FirstChildElement("point0")->Attribute("posy")), atof(child->FirstChildElement("point0")->Attribute("posz"))),
				Point(atof(child->FirstChildElement("point1")->Attribute("posx")), atof(child->FirstChildElement("point1")->Attribute("posy")), atof(child->FirstChildElement("point1")->Attribute("posz"))),
				Point(atof(child->FirstChildElement("point2")->Attribute("posx")), atof(child->FirstChildElement("point2")->Attribute("posy")), atof(child->FirstChildElement("point2")->Attribute("posz")))
			)
		);
	}

	return triangles;
}

std::vector<Camera> MyXMLReader::GetCameras()
{
	std::vector<Camera> cameras;

	tinyxml2::XMLElement *cameraElement = doc.FirstChildElement("root")->FirstChildElement("cameras");

	for (tinyxml2::XMLElement* child = cameraElement->FirstChildElement(); child; child = child->NextSiblingElement())
	{
		cameras.push_back(
			Camera(
				Point(atof(child->Attribute("posx")), atof(child->Attribute("posy")), atof(child->Attribute("posz")))));
	}

	return cameras;
}

