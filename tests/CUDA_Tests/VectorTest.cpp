#include "gtest\gtest.h"
#include "gmock\gmock.h"

#include "Vector.h"
#include "Exception.h"

TEST(Vector, CREATION){
	Vector v(Point(1, 2, 3), Point(0, 0, 1));

	Point p1(1, 2, 3);
	Point p2(0, 0, 1);

	ASSERT_EQ(p1, v.Location);
	ASSERT_EQ(p2, v.Direction);
	ASSERT_EQ(1, v.Length);
}

TEST(Vector, CTOR_THROWS_EXCEPTION) {
	ASSERT_THROW(Vector v(Point(1, 2, 3), Point(0, 1, 1)); , Exception);
}

TEST(Vector, GET_END_POINT) {
	Vector v(Point(1, 1, 1), Point(0, 0, 1));

	Point p = v.GetEndPoint();

	Point expected(1, 1, 2);
	ASSERT_EQ(expected, p);
}

TEST(Vector, EQUALS) {
	Vector v1(Point(1, 2, 3), Point(0, 0, 1));
	Vector v2(Point(1, 2, 3), Point(0, 0, 1));

	bool b = v1 == v2;

	ASSERT_EQ(true, b);

	Vector v3(Point(0, 2, 3), Point(0, 1, 0));
	b = v1 == v3;
	ASSERT_EQ(false, b);
}

TEST(Vector, ASSIGN) {
	Vector v1(Point(1, 2, 3), Point(0, 0, 1));
	Vector v2  = v1;

	ASSERT_EQ(v1, v2);
}