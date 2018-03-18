#include "gtest\gtest.h"
#include "gmock\gmock.h"

#include "MyXMLReader.h"

TEST(MyXMLReader, LOAD_PARSE) {
	MyXMLReader xmlReader("In.xml");
	bool s = xmlReader.doc.Error();
	ASSERT_EQ(false, s);
}

TEST(MyXMLReader, PARSE_LIGHTS) {
	MyXMLReader xmlReader("In.xml");

	std::vector<LightSource*> lights = xmlReader.GetLightSources();

	ASSERT_EQ(LightSource(Point(0, 0, 10), 10), *lights.front());
}

TEST(MyXMLReader, PARSE_TRIANGLES) {
	MyXMLReader xmlReader("In.xml");

	std::vector<Triangle*> triangles = xmlReader.GetTriangles();

	ASSERT_EQ(Triangle(Point(10, -10, 5), Point(10, 10, 5), Point(-10, 10, 5)), *triangles.front());
}

TEST(MyXMLReader, PARSE_CAMERAS) {
	MyXMLReader xmlReader("In.xml");

	std::vector<Camera*> cameras = xmlReader.GetCameras();

	ASSERT_EQ(Camera(Point(0, 0, 0)), *cameras.front());
}