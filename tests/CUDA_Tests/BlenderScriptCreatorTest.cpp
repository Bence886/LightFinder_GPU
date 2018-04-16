#include "gtest\gtest.h"
#include "gmock\gmock.h"

#include "BelnderScriptCreator.h"
#include "Point.h"

TEST(BlenderScriptCreator, CREATE_FILE){
	BlenderScriptCreator bsc("TestFile.txt");
}

TEST(BlenderScriptCreator, CREATE_OBJECT) {
	BlenderScriptCreator bsc("TestFile.txt");

	Point points[3];
	points[0] = (Point(1, 1, 1));
	points[1] = (Point(2, 2, 2));
	points[2] = (Point(0, 0, 0));

	bsc.CreateObject(points, "Teszt object", 3);
}