#include "gtest\gtest.h"
#include "gmock\gmock.h"

#include "BelnderScriptCreator.h"
#include "Point.h"

TEST(BlenderScriptCreator, CREATE_FILE){
	BlenderScriptCreator bsc("TestFile.txt");
}

TEST(BlenderScriptCreator, CREATE_OBJECT) {
	BlenderScriptCreator bsc("TestFile.txt");

	std::vector<Point> points;
	points.push_back(Point(0, 0, 0));
	points.push_back(Point(1, 1, 1));
	points.push_back(Point(2, 2, 2));

	bsc.CreateObject(points, "Teszt object");
}