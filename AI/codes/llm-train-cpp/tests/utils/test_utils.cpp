#include "llm/utils.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

TEST(Product, EmptyShapeReturnsOne) {
    EXPECT_EQ(product({}), 1);
}

TEST(Product, MultipliesAllDims) {
    EXPECT_EQ(product({2, 3, 4}), 24);
    EXPECT_EQ(product({7}), 7);
}

TEST(Product, ReturnsZeroWhenAnyDimZero) {
    EXPECT_EQ(product({5, 0, 7}), 0);
}

TEST(CanonicalDim, NonNegativeDimUnchanged) {
    EXPECT_EQ(canonical_dim(0, 3), 0);
    EXPECT_EQ(canonical_dim(2, 3), 2);
}

TEST(CanonicalDim, NegativeDimWrapsFromEnd) {
    EXPECT_EQ(canonical_dim(-1, 3), 2);
    EXPECT_EQ(canonical_dim(-3, 3), 0);
}

TEST(CanonicalDim, OutOfRangeThrows) {
    EXPECT_THROW(canonical_dim(3, 3), std::runtime_error);
    EXPECT_THROW(canonical_dim(-4, 3), std::runtime_error);
}
