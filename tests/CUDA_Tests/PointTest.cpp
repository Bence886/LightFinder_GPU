#include "gtest\gtest.h"
#include "gmock\gmock.h"

#include "Point.h"

TEST(Point, Creation) {
	Point P(1, 2, 3);

	ASSERT_EQ(1, P.X);
	ASSERT_EQ(2, P.Y);
	ASSERT_EQ(3, P.Z);
}

TEST(Point, ADD) {
	Point p1(1, 2, 3);
	Point p2(1, 1, 1);

	Point p3 = p1 + p2;

	ASSERT_EQ(2, p3.X);
	ASSERT_EQ(3, p3.Y);
	ASSERT_EQ(4, p3.Z);
}

TEST(Point, SUBSTRACT){
	Point p1(1, 2, 3);
	Point p2(1, 1, 1);

	Point p3 = p1 - p2;

	ASSERT_EQ(0, p3.X);
	ASSERT_EQ(1, p3.Y);
	ASSERT_EQ(2, p3.Z);
}

TEST(Point, EQUALS) {
	Point p1(1, 2, 3);
	Point p2(1, 2, 3);

	bool eq = p1 == p2;
	ASSERT_EQ(true, eq);
}

TEST(Point, NOT_EQUALS) {
	Point p1(1, 2, 3);
	Point p2(1, 1, 3);

	bool eq = p1 == p2;
	ASSERT_EQ(false, eq);
}

TEST(POINT, ASSIGN) {
	Point p1(1, 2, 3);
	Point p2 = p1;

	ASSERT_EQ(1, p2.X);
	ASSERT_EQ(2, p2.Y);
	ASSERT_EQ(3, p2.Z);
}

TEST(Point, MULTIPLY_BY_LAMBDA) {
	Point p(1, 2, 3);
	p.MultipyByLambda(2);

	ASSERT_EQ(2, p.X);
	ASSERT_EQ(4, p.Y);
	ASSERT_EQ(6, p.Z);
}

TEST(Point, DEVIDE_BY_LAMBDA){
	Point p(2, 4, 6);
	p.DevideByLambda(2);

	ASSERT_EQ(1, p.X);
	ASSERT_EQ(2, p.Y);
	ASSERT_EQ(3, p.Z);
}