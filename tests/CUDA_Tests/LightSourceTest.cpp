#include "gtest\gtest.h"
#include "gmock\gmock.h"

#include "LightSource.h"

TEST(LithSource, EQUALS) {
	LightSource ls0(Point(0, 1, -456), 10);
	LightSource ls1(Point(0, 1, -456), 10);

	ASSERT_EQ(ls0, ls1);
}