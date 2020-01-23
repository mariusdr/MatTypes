#include <gtest/gtest.h>
#include <type_traits>
#include <tuple>

#include "Types.hpp"


template <typename T>
class VectorTester : public ::testing::Test {};

using test_types = ::testing::Types< 
    std::integral_constant<std::size_t, 2>,
    std::integral_constant<std::size_t, 3>,
    std::integral_constant<std::size_t, 4>,
    std::integral_constant<std::size_t, 5>,
    std::integral_constant<std::size_t, 6>,
    std::integral_constant<std::size_t, 7>,
    std::integral_constant<std::size_t, 8>,
    std::integral_constant<std::size_t, 9>
>;


TYPED_TEST_CASE(VectorTester, test_types);

TYPED_TEST(VectorTester, test_size)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x;
    ASSERT_EQ(x.size(), Len);
}

TYPED_TEST(VectorTester, test_addition)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x(12.3);
    mt::Vector<Len> y(23.1);
    mt::Vector<Len> z = x + y;

    for (size_t i = 0; i < Len; ++i)
    {
        EXPECT_FLOAT_EQ(z.data[i], 12.3 + 23.1);
    }
}

TYPED_TEST(VectorTester, test_scalar_mult)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x(12.3);
    mt::Vector<Len> y = x * 4.2;
    mt::Vector<Len> z = 4.2 * x;
    
    for (size_t i = 0; i < Len; ++i)
    {
        EXPECT_FLOAT_EQ(y.data[i], 4.2 * 12.3);
        EXPECT_FLOAT_EQ(z.data[i], 4.2 * 12.3);
    }
}

TYPED_TEST(VectorTester, test_scalar_div)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x(12.3);
    mt::Vector<Len> y = x / 4.2;
    
    for (size_t i = 0; i < Len; ++i)
    {
        EXPECT_FLOAT_EQ(y.data[i], 12.3 / 4.2);
    }
}

TYPED_TEST(VectorTester, test_length)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x;

    float ss = 0.f;
    for (int i = 0; i < Len; ++i)
    {
        float f = 0.2 * i;
        x.data[i] = f;
        ss += (f*f);
    }
    EXPECT_FLOAT_EQ(sqrt(ss), x.length());
}

TYPED_TEST(VectorTester, dot_product_orthogonal)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x = mt::zeros<Len>();
    mt::Vector<Len> y = mt::zeros<Len>();

    if (Len > 2)
    {
        x.data[Len - 2] = 1.f;
        y.data[Len - 1] = 1.f;

        EXPECT_FLOAT_EQ(x.dot(y), 0.f);
    }
}

TYPED_TEST(VectorTester, dot_product_codirectional)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x = mt::zeros<Len>();
    mt::Vector<Len> y = mt::zeros<Len>();

    if (Len > 2)
    {
        x.data[Len - 1] = 12.f;
        y.data[Len - 1] = 1.f;

        EXPECT_FLOAT_EQ(x.dot(y), 12.f * 1.f);
    }
}

TYPED_TEST(VectorTester, dot_product_antidirectional)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x = mt::zeros<Len>();
    mt::Vector<Len> y = mt::zeros<Len>();

    if (Len > 2)
    {
        x.data[Len - 1] = 12.f;
        y.data[Len - 1] = -1.f;

        EXPECT_FLOAT_EQ(x.dot(y), -12.f * 1.f);
    }
}

TYPED_TEST(VectorTester, length_and_dot_prod)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x;

    for (int i = 0; i < Len; ++i)
    {
        float f = 0.2 * i;
        x.data[i] = f;
    }

    float d = sqrt(x.dot(x));
    ASSERT_FLOAT_EQ(d, x.length());
}

TYPED_TEST(VectorTester, approx_eq)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x;
    mt::Vector<Len> y;

    for (int i = 0; i < Len; ++i)
    {
        float f = 0.2 * i + 0.01;
        x.data[i] = f;
        y.data[i] = 0.002 * f;
    }
    EXPECT_TRUE(x.approx_equal(x));
    EXPECT_FALSE(x.approx_equal(y));
}

TYPED_TEST(VectorTester, abs)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x;

    for (int i = 0; i < Len; ++i)
    {
        float f = 0.2 * i;
        if (i % 2 == 0)
        {
            x.data[i] = f;
        }
        else 
        {
            x.data[i] = -f;
        }
    }

    for (int i = 0; i < Len; ++i)
    {
        EXPECT_TRUE(x.abs().data[i] >= 0.f);
    }
}

TYPED_TEST(VectorTester, normalized)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x;

    for (int i = 0; i < Len; ++i)
    {
        float f = 0.2 * i;
        if (i % 2 == 0)
        {
            x.data[i] = f;
        }
        else 
        {
            x.data[i] = -f;
        }
    }
    
    mt::Vector<Len> xn = x.normalized();

    // length should be one 
    EXPECT_FLOAT_EQ(xn.length(), 1.f);

    // direction should be the same as x
    EXPECT_FLOAT_EQ(xn.dot(x), x.length());
}

TYPED_TEST(VectorTester, test_zero_const)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x = mt::zeros<Len>();

    for (int i = 0; i < Len; ++i)
    {
        EXPECT_FLOAT_EQ(x.data[i], 0.f);
    }
}

TYPED_TEST(VectorTester, test_one_const)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x = mt::ones<Len>();

    for (int i = 0; i < Len; ++i)
    {
        EXPECT_FLOAT_EQ(x.data[i], 1.f);
    }
}

TYPED_TEST(VectorTester, test_float_cat)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x = mt::ones<Len>();

    mt::Vector<Len+1> lx = mt::cat(0.f, x);
    mt::Vector<Len+1> rx = mt::cat(x, 0.f); 

    EXPECT_EQ(lx.size(), Len + 1);
    EXPECT_EQ(rx.size(), Len + 1);

    EXPECT_FLOAT_EQ(lx.data[0], 0.f);
    for (int i = 1; i < Len - 1; ++i)
    {
        EXPECT_FLOAT_EQ(rx.data[i], 1.f);
        EXPECT_FLOAT_EQ(lx.data[i], 1.f);
    }
    EXPECT_FLOAT_EQ(rx.data[Len], 0.f);
}

TYPED_TEST(VectorTester, test_vect_cat)
{
    static constexpr std::size_t Len = TypeParam::value;
    mt::Vector<Len> x = mt::ones<Len>();

    mt::Vector<5> y = mt::zeros<5>();

    mt::Vector<Len + 5> lz = mt::cat(y, x);
    mt::Vector<Len + 5> rz = mt::cat(x, y);

    EXPECT_EQ(lz.size(), Len + 5);
    EXPECT_EQ(rz.size(), Len + 5);

    for (int i = 0; i < Len + 5; ++i)
    {
        if (i < 5)
        {
            EXPECT_EQ(lz.data[i], 0.f);
        }
        else 
        {
            EXPECT_EQ(lz.data[i], 1.f);
        }
    }
    for (int i = 0; i < Len + 5; ++i)
    {
        if (i < Len)
        {
            EXPECT_EQ(rz.data[i], 1.f);
        }
        else 
        {
            EXPECT_EQ(rz.data[i], 0.f);
        }
    }
}


//===================================================================//

class Vector3Tester : public ::testing::Test {};

TEST(Vector3Tester, dot)
{
    mt::Vector3f x = mt::vec3f(1.2, 4.2, -1.9);
    mt::Vector3f y = mt::vec3f(-9.4, 1.0, 5.5);

    float a = x.dot(y);
    float b = y.dot(x);

    EXPECT_FLOAT_EQ(a, b);
    EXPECT_FLOAT_EQ(a, -17.53);
}

TEST(Vector3Tester, dot_codirectional)
{
    mt::Vector3f x = mt::vec3f(0, 9, 0);
    mt::Vector3f y = mt::vec3f(0, 9, 0);

    float a = x.dot(y);
    float b = y.dot(x);

    EXPECT_FLOAT_EQ(a, b);
    EXPECT_FLOAT_EQ(a, 9*9);
}

TEST(Vector3Tester, dot_orthogonal)
{
    mt::Vector3f x = mt::vec3f(0, 9, 0);
    mt::Vector3f y = mt::vec3f(8, 0, 0);

    float a = x.dot(y);
    float b = y.dot(x);

    EXPECT_FLOAT_EQ(a, b);
    EXPECT_FLOAT_EQ(a, 0);
}

TEST(Vector3Tester, cross)
{
    mt::Vector3f x = mt::vec3f(1.2, 4.2, -1.9);
    mt::Vector3f y = mt::vec3f(-9.4, 1.0, 5.5);

    mt::Vector3f z = mt::cross(x, y);

    EXPECT_FLOAT_EQ(z.data[0], 25);
    EXPECT_FLOAT_EQ(z.data[1], 11.26);
    EXPECT_FLOAT_EQ(z.data[2], 40.68);

    mt::Vector3f w = mt::cross(y, x);

    EXPECT_FLOAT_EQ(w.data[0], -25);
    EXPECT_FLOAT_EQ(w.data[1], -11.26);
    EXPECT_FLOAT_EQ(w.data[2], -40.68);

    x = mt::vec3f(1, 0, 0);
    y = mt::vec3f(0, 1, 0);
    z = mt::vec3f(0, 0, 1);

    EXPECT_TRUE(z.approx_equal(mt::cross(x, y)));
    EXPECT_TRUE(y.approx_equal(mt::cross(z, x)));
    EXPECT_TRUE(x.approx_equal(mt::cross(y, z)));
}

TEST(Vector3Tester, deg_rad_conversions)
{
    using P = std::pair<float, float>;

    std::vector<P> ps 
    {
        P(0, 0), P(90, 0.5*M_PI), P(180, M_PI), 
        P(270, 1.5*M_PI), P(360, 2*M_PI),
        P(0, 0), P(-90, -0.5*M_PI), P(-180, -M_PI), 
        P(-270, -1.5*M_PI), P(-360, -2*M_PI)
    };

    for (P p: ps)
    {
        float deg = p.first;
        float rad = p.second;
        EXPECT_FLOAT_EQ(mt::deg_to_rad(deg), rad);
        EXPECT_FLOAT_EQ(mt::rad_to_deg(rad), deg);
    }
}

//===================================================================//

template <typename T>
class SquareMatrixTester : public ::testing::Test {};

TYPED_TEST_CASE(SquareMatrixTester, test_types);



//===================================================================//

template <typename T>
class MatrixTester : public ::testing::Test {};

using test_types_s = ::testing::Types< 
    std::integral_constant<std::size_t, 606>,
    std::integral_constant<std::size_t, 303>,
    std::integral_constant<std::size_t, 404>,
    std::integral_constant<std::size_t, 203>,
    std::integral_constant<std::size_t, 302>,
    std::integral_constant<std::size_t, 402>,
    std::integral_constant<std::size_t, 204>,
    std::integral_constant<std::size_t, 304>,
    std::integral_constant<std::size_t, 403>,
    std::integral_constant<std::size_t, 507>,
    std::integral_constant<std::size_t, 705>,
    std::integral_constant<std::size_t, 207>,
    std::integral_constant<std::size_t, 702>,
    std::integral_constant<std::size_t, 509>,
    std::integral_constant<std::size_t, 905>
>;


TYPED_TEST_CASE(MatrixTester, test_types_s);

TYPED_TEST(MatrixTester, shape)
{
    static constexpr std::size_t Rows = TypeParam::value / 100;
    static constexpr std::size_t Cols = TypeParam::value % 100;

    mt::Matrix<Rows, Cols> m;
    EXPECT_EQ(m.rows(), Rows);
    EXPECT_EQ(m.cols(), Cols);
}

TYPED_TEST(MatrixTester, rows)
{
    static constexpr std::size_t Rows = TypeParam::value / 100;
    static constexpr std::size_t Cols = TypeParam::value % 100;

    mt::Matrix<Rows, Cols> mat;
    mt::Vector<Cols> r(1.23);

    for (int i = 0; i < Rows; ++i)
    {
        mat.set_row(i, r);
        EXPECT_TRUE(r.approx_equal(mat.get_row(i)));
    }
}

TYPED_TEST(MatrixTester, cols)
{
    static constexpr std::size_t Rows = TypeParam::value / 100;
    static constexpr std::size_t Cols = TypeParam::value % 100;

    mt::Matrix<Rows, Cols> mat;
    mt::Vector<Rows> r(1.23);

    for (int i = 0; i < Cols; ++i)
    {
        mat.set_col(i, r);
        EXPECT_TRUE(r.approx_equal(mat.get_col(i)));
    }
}

TYPED_TEST(MatrixTester, transpose)
{
    static constexpr std::size_t Rows = TypeParam::value / 100;
    static constexpr std::size_t Cols = TypeParam::value % 100;

    mt::Matrix<Rows, Cols> m;
    for (int i = 0; i < Rows; ++i)
    {
        for (int j = 0; j < Cols; ++j)
        {
            m.data[i * Cols + j] = 1.2f * float(i) + 0.2f * float(j);
        }
    }

    EXPECT_EQ(m.transpose().rows(), Cols);
    EXPECT_EQ(m.transpose().cols(), Rows);

    mt::Matrix<Cols, Rows> mt = m.transpose();

    for (int i = 0; i < Rows; ++i)
    {
        mt::Vector<Cols> x = m.get_row(i);
        mt::Vector<Cols> xt = mt.get_col(i);

        EXPECT_TRUE(x.approx_equal(xt));
    }
    
    for (int j = 0; j < Cols; ++j)
    {
        mt::Vector<Rows> x = m.get_col(j);
        mt::Vector<Rows> xt = mt.get_row(j);

        EXPECT_TRUE(x.approx_equal(xt));
    }
}




//===================================================================//

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}