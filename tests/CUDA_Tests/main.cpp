#include "gtest\gtest.h"
#include "gmock\gmock.h"

#include "Point.h"

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}