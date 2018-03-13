#include "gtest\gtest.h"
#include "gmock\gmock.h"

#include "Point.h"

#define FLOAT_PRECISION 0.00000001

TEST(Point, CREATE) {
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

TEST(Point, ASSIGN) {
	Point p1(1, 2, 3);
	Point p2 = p1;

	ASSERT_EQ(1, p2.X);
	ASSERT_EQ(2, p2.Y);
	ASSERT_EQ(3, p2.Z);
}

TEST(Point, MULTIPLY_BY_LAMBDA) {
	Point p(1, 2, 3);
	p.MultiplyByLambda(2);

	ASSERT_EQ(2, p.X);
	ASSERT_EQ(4, p.Y);
	ASSERT_EQ(6, p.Z);
}

TEST(Point, DEVIDE_BY_LAMBDA){
	Point p(1, 4, 6);
	p.DevideByLambda(2);

	ASSERT_EQ(0.5, p.X);
	ASSERT_EQ(2, p.Y);
	ASSERT_EQ(3, p.Z);
}

TEST(Point, NORMALIZE){
	Point p(3, 1, 2);
	p.Normalize();

	ASSERT_NEAR(0.801783681, p.X, FLOAT_PRECISION);
	ASSERT_NEAR(0.267261237, p.Y, FLOAT_PRECISION);
	ASSERT_NEAR(0.534522474, p.Z, FLOAT_PRECISION);
}

TEST(Point, LENGTH) {
	Point p(1, 2, 3);
	float l = p.Length();

	ASSERT_NEAR(3.74165750, l, FLOAT_PRECISION);
}

TEST(Point, GET_MIDLE_POINT) {
	Point p1(0, 0, 0);
	Point p2(1, 0, 0);

	Point p3 = Point::GetMidlePoint(p1, p2);

	ASSERT_EQ(0.5, p3.X);
	ASSERT_EQ(0, p3.Y);
	ASSERT_EQ(0, p3.Z);
}

TEST(Point, DOT_PRODUCT) {
	Point p1(1, 3, -5);
	Point p2(4, -2, -1);

	float d = Point::DotProduct(p1, p2);

	ASSERT_EQ(3, d);
}

TEST(Point, CROSS_PRODUCT) {
	Point p1(2, 3, 4);
	Point p2(5, 6, 7);

	Point p = Point::CrossProduct(p1, p2);

	ASSERT_EQ(-3, p.X);
	ASSERT_EQ(6, p.Y);
	ASSERT_EQ(-3, p.Z);
}

TEST(Point, DISTANCE) {
	Point p1(3, 5, 7);
	Point p2(8, 2, 6);

	float f = Point::Distance(p1, p2);

	ASSERT_NEAR(5.91608, f, FLOAT_PRECISION);
}