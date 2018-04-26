#include "gtest\gtest.h"
#include "gmock\gmock.h"

#include "Triangle.h"
#include "Point.h"
#include "Vector.h"

TEST(Triangle, CREATE) {
	Triangle t(Point(0, 0, 0), Point(-50, 134, 34), Point(1, 2, 3));

	Point p0(0, 0, 0);
	Point p1(-50, 134, 34);
	Point p2(1, 2, 3);

	ASSERT_EQ(p0, t.p0);
	ASSERT_EQ(p1, t.p1);
	ASSERT_EQ(p2, t.p2);

	ASSERT_EQ(1, 1);
}

TEST(Triangle, NORMAL_0) {
	//arrange
	Point p0(1, -1, 0);
	Point p1(1, 1, 0);
	Point p2(-1, 1, 0);
	Triangle T(p0, p1, p2);
	Point V(0, 0, 1);

	//act
	Point N = T.normal;

	//assert
	ASSERT_EQ(V.X, N.X);
	ASSERT_EQ(V.Y, N.Y);
	ASSERT_EQ(V.Z, N.Z);
}

TEST(Triangle, NORMAL_1) {
	//arrange
	Point p0(-1, 0, 0);
	Point p1(0, 0, 1);
	Point p2(1, 0, 0);
	Triangle T(p0, p1, p2);
	Point V(0, 1, 0);

	//act
	Point N = T.normal;

	//assert
	ASSERT_EQ(V.X, N.X);
	ASSERT_EQ(V.Y, N.Y);
	ASSERT_EQ(V.Z, N.Z);
}

TEST(Triangle, InsideTriangle) {
	Point p0(10, -10, 3);
	Point p1(10, 10, 3);
	Point p2(-10, -10, 3);
	Triangle T(p0, p1, p2);
	Point p3(5, -5, 10);
	Point p4(0, 0, -1);

	Vector V(p3, p4);
	Point R(5, -5, 3);

	Point H = *T.InsideTriangle(V);

	ASSERT_EQ(R, H);
}

TEST(Triangle, EQUALS) {
	Point p0(10, -10, 3);
	Point p1(10, 10, 3);
	Point p2(-10, -10, 3);
	Triangle T1(p0, p1, p2);

	Point p3(10, -10, 3);
	Point p4(10, 10, 3);
	Point p5(-10, -10, 3);
	Triangle T2(p3, p4, p5);

	ASSERT_EQ(T1, T2);
}