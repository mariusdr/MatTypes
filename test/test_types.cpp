#include <gtest/gtest.h>
#include <type_traits>
#include <tuple>

#include "../src/math_types.hpp"

using namespace cumanip;


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
    std::integral_constant<std::size_t, 9>,
    std::integral_constant<std::size_t, 24>,
    std::integral_constant<std::size_t, 30>,
    std::integral_constant<std::size_t, 99>
    // std::integral_constant<std::size_t, 256>,
    // std::integral_constant<std::size_t, 512>
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

class Matrix3Tester : public ::testing::Test {};

TEST(Matrix3Tester, mult)
{
    mt::Matrix3f a = mt::mat3f(
        0.1, 8.2, 7.7,
        8.5, -1.2, 9.99,
        0.01, 5.4, 9.2
    );

    mt::Matrix3f b = mt::mat3f(
        6.6, -2.11, 0.314,
        4.2, 13.37, 0.09,
        8.4, 3.2, 2.2
    );

    mt::Matrix3f ab = a * b;
    mt::Matrix3f ba = b * a;

    mt::Matrix3f ab_t = mt::mat3f(
        99.78, 134.063, 17.7094,
        134.976, -2.011, 24.539,
        100.026, 101.6169, 20.72914
    );

    mt::Matrix3f ba_t = mt::mat3f(
        -17.27186, 58.3476, 32.6299,
        114.0659, 18.882, 166.7343,
        28.062, 76.92, 116.888
    );

    for (int i = 0; i < 9; ++i)
    {
        ASSERT_FLOAT_EQ(ab.data[i], ab_t.data[i]);
        ASSERT_FLOAT_EQ(ba.data[i], ba_t.data[i]);
    }
    
    EXPECT_TRUE(ab.approx_equal(ab_t));
    EXPECT_TRUE(ab_t.approx_equal(ab));

    EXPECT_TRUE(ba.approx_equal(ba_t));
    EXPECT_TRUE(ba_t.approx_equal(ba));

    EXPECT_FALSE(ab.approx_equal(ba));
    EXPECT_FALSE(ba.approx_equal(ab));
}

TEST(Matrix3Tester, from_roll)
{
    mt::Vector3f x = mt::vec3f(1, 0, 0);
    mt::Vector3f y = mt::vec3f(0, 1, 0);
    mt::Vector3f z = mt::vec3f(0, 0, 1);

    float rx = 0.162;
    mt::Matrix3f rot = mt::from_roll(rx);

    auto xt = rot * x;
    auto yt = rot * y;
    auto zt = rot * z;

    // dont transform the rotation axis
    EXPECT_TRUE(xt.approx_equal(x));

    // transformed axes should still be orthogonal
    EXPECT_FLOAT_EQ(xt.dot(yt), 0.f);
    EXPECT_FLOAT_EQ(xt.dot(zt), 0.f);
    EXPECT_FLOAT_EQ(yt.dot(zt), 0.f);

    // righthanded system should still be righthanded
    EXPECT_TRUE((mt::cross(xt, yt)).approx_equal(zt));
    EXPECT_TRUE((mt::cross(yt, zt)).approx_equal(xt));
    EXPECT_TRUE((mt::cross(zt, xt)).approx_equal(yt));

    // rotations should be self inverse
    mt::Matrix3f id = mt::identity<3, 3>();
    EXPECT_TRUE((rot * rot.transpose()).approx_equal(id));
}


TEST(Matrix3Tester, from_pitch)
{
    mt::Vector3f x = mt::vec3f(1, 0, 0);
    mt::Vector3f y = mt::vec3f(0, 1, 0);
    mt::Vector3f z = mt::vec3f(0, 0, 1);

    float rx = 8.862;
    mt::Matrix3f rot = mt::from_pitch(rx);

    auto xt = rot * x;
    auto yt = rot * y;
    auto zt = rot * z;

    // dont transform the rotation axis
    EXPECT_TRUE(yt.approx_equal(y));

    // transformed axes should still be orthogonal
    EXPECT_FLOAT_EQ(xt.dot(yt), 0.f);
    EXPECT_FLOAT_EQ(xt.dot(zt), 0.f);
    EXPECT_FLOAT_EQ(yt.dot(zt), 0.f);
    
    // righthanded system should still be righthanded
    EXPECT_TRUE((mt::cross(xt, yt)).approx_equal(zt));
    EXPECT_TRUE((mt::cross(yt, zt)).approx_equal(xt));
    EXPECT_TRUE((mt::cross(zt, xt)).approx_equal(yt));

    // rotations should be self inverse
    mt::Matrix3f id = mt::identity<3, 3>();
    EXPECT_TRUE((rot * rot.transpose()).approx_equal(id));
}

TEST(Matrix3Tester, from_yaw)
{
    mt::Vector3f x = mt::vec3f(1, 0, 0);
    mt::Vector3f y = mt::vec3f(0, 1, 0);
    mt::Vector3f z = mt::vec3f(0, 0, 1);

    float rx = 7.162;
    mt::Matrix3f rot = mt::from_yaw(rx);

    auto xt = rot * x;
    auto yt = rot * y;
    auto zt = rot * z;

    // dont transform the rotation axis
    EXPECT_TRUE(zt.approx_equal(z));

    // transformed axes should still be orthogonal
    EXPECT_FLOAT_EQ(xt.dot(yt), 0.f);
    EXPECT_FLOAT_EQ(xt.dot(zt), 0.f);
    EXPECT_FLOAT_EQ(yt.dot(zt), 0.f);
    
    // righthanded system should still be righthanded
    EXPECT_TRUE((mt::cross(xt, yt)).approx_equal(zt));
    EXPECT_TRUE((mt::cross(yt, zt)).approx_equal(xt));
    EXPECT_TRUE((mt::cross(zt, xt)).approx_equal(yt));

    // rotations should be self inverse
    mt::Matrix3f id = mt::identity<3, 3>();
    EXPECT_TRUE((rot * rot.transpose()).approx_equal(id));
}

TEST(Matrix3Tester, combining_rotations)
{
    // rotations around the same axis should add up
    EXPECT_TRUE(mt::from_roll(0.12).approx_equal(mt::from_roll(0.07) * mt::from_roll(0.05)));
    EXPECT_TRUE(mt::from_pitch(0.12).approx_equal(mt::from_pitch(0.07) * mt::from_pitch(0.05)));
    EXPECT_TRUE(mt::from_yaw(0.12).approx_equal(mt::from_yaw(0.07) * mt::from_yaw(0.05)));

    auto a = mt::from_roll(2.31);
    auto b = mt::from_pitch(3.22);
    auto c = mt::from_yaw(0.991);

    // rotations should not commute generally
    EXPECT_FALSE((a * b).approx_equal(b * a));
    EXPECT_FALSE((a * c).approx_equal(c * a));
    EXPECT_FALSE((b * c).approx_equal(c * b));

    // rotations should be self inverse
    mt::Matrix3f id = mt::identity<3, 3>();
    {
        auto ab = a * b;
        auto abt = ab.transpose();
        auto ba = b * a;
        auto bat = ba.transpose();
        auto aba = a * b * a;
        auto abat = aba.transpose();
        auto bba = b * b * a;
        auto bbat = bba.transpose();
        auto bbc = b * b * c;
        auto bbct = bbc.transpose();
        auto ac = a * c;
        auto act = ac.transpose();
        EXPECT_TRUE((ab * abt).approx_equal(id));
        EXPECT_TRUE((ba * bat).approx_equal(id));
        EXPECT_TRUE((aba * abat).approx_equal(id));
        EXPECT_TRUE((bba * bbat).approx_equal(id));
        EXPECT_TRUE((bbc * bbct).approx_equal(id));
        EXPECT_TRUE((ac * act).approx_equal(id));
        EXPECT_TRUE((abt * ab).approx_equal(id));
        EXPECT_TRUE((bat * ba).approx_equal(id));
        EXPECT_TRUE((abat * aba).approx_equal(id));
    }

    // test fromRPY and fromYPR
    auto abc = mt::fromRPY(2.31, 3.22, 0.991);
    auto cba = mt::fromYPR(0.991, 3.22, 2.31);

    EXPECT_TRUE(abc.approx_equal(c * b * a));
    EXPECT_TRUE((c * b * a).approx_equal(abc));
    EXPECT_TRUE(cba.approx_equal(a * b * c));
    EXPECT_TRUE((a * b * c).approx_equal(cba));

    EXPECT_TRUE((abc * abc.transpose()).approx_equal(id));
    EXPECT_TRUE((abc.transpose() * abc).approx_equal(id));
    EXPECT_TRUE((cba * cba.transpose()).approx_equal(id));
    EXPECT_TRUE((cba.transpose() * cba).approx_equal(id));

    EXPECT_FALSE(abc.approx_equal(cba));


    mt::Vector3f x = mt::vec3f(1, 0, 0);
    mt::Vector3f y = mt::vec3f(0, 1, 0);
    mt::Vector3f z = mt::vec3f(0, 0, 1);

    {
        auto xt = abc * x;
        auto yt = abc * y;
        auto zt = abc * z;

        // transformed axes should still be orthogonal
        EXPECT_NEAR(xt.dot(yt), 0.f, 1e-6);
        EXPECT_NEAR(xt.dot(zt), 0.f, 1e-6);
        EXPECT_NEAR(yt.dot(zt), 0.f, 1e-6);

        // righthanded system should still be righthanded
        EXPECT_TRUE((mt::cross(xt, yt)).approx_equal(zt));
        EXPECT_TRUE((mt::cross(yt, zt)).approx_equal(xt));
        EXPECT_TRUE((mt::cross(zt, xt)).approx_equal(yt));
    }
        
    {
        auto xt = cba * x;
        auto yt = cba * y;
        auto zt = cba * z;

        // transformed axes should still be orthogonal
        EXPECT_NEAR(xt.dot(yt), 0.f, 1e-6);
        EXPECT_NEAR(xt.dot(zt), 0.f, 1e-6);
        EXPECT_NEAR(yt.dot(zt), 0.f, 1e-6);

        // righthanded system should still be righthanded
        EXPECT_TRUE((mt::cross(xt, yt)).approx_equal(zt));
        EXPECT_TRUE((mt::cross(yt, zt)).approx_equal(xt));
        EXPECT_TRUE((mt::cross(zt, xt)).approx_equal(yt));
    }

    auto v1 = mt::vec3f(93.11, -17.2, 0.08).normalized();
    auto v2 = mt::vec3f(-5.5, 2.3, -9.1).normalized();

    // angle between two rotated vectors should not change 
    {
        auto v1t = abc * v1;
        auto v2t = abc * v2;
        EXPECT_FLOAT_EQ(mt::angle(v1, v2), mt::angle(v1t, v2t));
    }

    {
        auto v1t = cba * v1;
        auto v2t = cba * v2;
        EXPECT_FLOAT_EQ(mt::angle(v1, v2), mt::angle(v1t, v2t));
    }

    // length of rotated vector should not change
    auto v3 = mt::vec3f(9.1, -1.111, 8.888);

    {
        auto v3t = abc * v3;
        EXPECT_FLOAT_EQ(v3.length(), v3t.length());
    }
    {
        auto v3t = cba * v3;
        EXPECT_FLOAT_EQ(v3.length(), v3t.length());
    }
}


TEST(Matrix3Tester, to_rpy)
{
    mt::Matrix3f rot = mt::fromRPY(0.31, 1.42, -0.2);
    auto rpy = mt::toRPY(rot);
    mt::Matrix3f rot2 = mt::fromRPY(rpy);

    // rot and rot2 should be equivalent rotations
    
    mt::Vector3f x = mt::vec3f(1, 0, 0);
    mt::Vector3f y = mt::vec3f(0, 1, 0);
    mt::Vector3f z = mt::vec3f(0, 0, 1);

    EXPECT_TRUE((rot * x).approx_equal(rot2 * x));
    EXPECT_TRUE((rot * y).approx_equal(rot2 * y));
    EXPECT_TRUE((rot * z).approx_equal(rot2 * z));

    mt::Vector3f v1 = mt::vec3f(0.31, -9.912, 0.127);
    mt::Vector3f v2 = mt::vec3f(-2.11, 0.31, 4.20);

    EXPECT_TRUE((rot * v1).approx_equal(rot2 * v1));
    EXPECT_TRUE((rot * v2).approx_equal(rot2 * v2));

    // one of both solutions of fromRPY should be equal to the original euler angles

    // these angles are non singular, so they should be recovered fine
    std::vector<mt::Vector3f> tgs =
    {
        mt::vec3f(1.42, 0.4, -1.2),
        mt::vec3f(-1.42, 0.4, -1.2),
        mt::vec3f(0.21, 0.224, 0.112),
        mt::vec3f(1.2, 1.4, 1.6),
        mt::vec3f(0.4, 0.2, 0),
    };
    for (auto& t: tgs)
    {
        auto s1 = mt::toRPY(mt::fromRPY(t), 0);
        auto s2 = mt::toRPY(mt::fromRPY(t), 1);
        const float s1_r = fabs(s1.data[0]-t.data[0]);
        const float s1_p = fabs(s1.data[1]-t.data[1]);
        const float s1_y = fabs(s1.data[2]-t.data[2]);
        const float s2_r = fabs(s2.data[0]-t.data[0]);
        const float s2_p = fabs(s2.data[1]-t.data[1]);
        const float s2_y = fabs(s2.data[2]-t.data[2]);

        EXPECT_TRUE(((s1_r < 1e-6) && (s1_p < 1e-6) && (s1_y < 1e-6)) || ((s2_r < 1e-6) && (s2_p < 1e-6) && (s2_y < 1e-6)));

        // both solutions should be equivalent rotations
        auto m1 = mt::fromRPY(t);
        auto ms1 = mt::fromRPY(s1);
        auto ms2 = mt::fromRPY(s2);

        EXPECT_TRUE((m1 * v1).approx_equal(ms1 * v1));
        EXPECT_TRUE((m1 * v2).approx_equal(ms1 * v2));
        EXPECT_TRUE((m1 * v1).approx_equal(ms2 * v1));
        EXPECT_TRUE((m1 * v2).approx_equal(ms2 * v2));
    }

    // // these angles have a gimbal lock, so we get singular solutions 
    // std::vector<mt::Vector3f> singular_tgs =
    // {
    //     mt::vec3f(M_PI_2, M_PI_2, M_PI_2),
    //     mt::vec3f(M_PI_2, M_PI_2, M_PI_2),
    //     mt::vec3f(M_PI_2, M_PI, M_PI_2),
    //     mt::vec3f(M_PI, M_PI_2, M_PI),
    //     mt::vec3f(-M_PI, M_PI_2, M_PI),
    //     mt::vec3f(M_PI, M_PI_2, -M_PI),
    //     mt::vec3f(M_PI, -M_PI_2, M_PI),
    // };
    
    // for (auto& t: tgs)
    // {
    //     auto s1 = mt::toRPY(mt::fromRPY(t), 0);
    //     auto s2 = mt::toRPY(mt::fromRPY(t), 1);
    //     EXPECT_FLOAT_EQ(s1.data[2], 0.f);
    //     EXPECT_FLOAT_EQ(s2.data[2], 0.f);
    // }
}


//===================================================================//


template <typename T>
class SquareMatrixTester : public ::testing::Test {};

TYPED_TEST_CASE(SquareMatrixTester, test_types);

TYPED_TEST(SquareMatrixTester, identity)
{
    static constexpr std::size_t N = TypeParam::value;

    mt::Matrix<N, N> id = mt::identity<N, N>();

    EXPECT_FLOAT_EQ(id.trace(), N);
    EXPECT_TRUE(id.approx_equal(id.transpose()));
}

TYPED_TEST(SquareMatrixTester, mat_mult)
{
    static constexpr std::size_t N = TypeParam::value;

    mt::Matrix<N, N> a(0.1);
    mt::Matrix<N, N> b(0.2);

    EXPECT_EQ((a*b).rows(), N);
    EXPECT_EQ((b*a).rows(), N);
    EXPECT_EQ((a*b).cols(), N);
    EXPECT_EQ((b*a).cols(), N);

    mt::Matrix<N, N> id = mt::identity<N, N>();
    EXPECT_TRUE((a*id).approx_equal(a));
    EXPECT_TRUE((id*a).approx_equal(a));
}

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

TYPED_TEST(MatrixTester, basic_ops)
{
    static constexpr std::size_t Rows = TypeParam::value / 100;
    static constexpr std::size_t Cols = TypeParam::value % 100;

    mt::Matrix<Rows, Cols> a(0.2f);

    auto b = 13.37 * a;
    auto c = a * 13.37;

    EXPECT_EQ(c.cols(), Cols);
    EXPECT_EQ(b.cols(), Cols);
    EXPECT_EQ(c.rows(), Rows);
    EXPECT_EQ(b.rows(), Rows);
    EXPECT_TRUE(b.approx_equal(c));
    EXPECT_TRUE(c.approx_equal(b));
    EXPECT_FLOAT_EQ(b.data[0], 0.2f * 13.37);
    EXPECT_FLOAT_EQ(c.data[0], 0.2f * 13.37);
    EXPECT_TRUE((a + b).approx_equal(b + a));
    EXPECT_FALSE((a - b).approx_equal(b - a));
    EXPECT_TRUE((a - a).approx_equal(mt::m_zeros<Rows, Cols>()));

    EXPECT_TRUE((a + b - b).approx_equal(a));
    EXPECT_TRUE((a + b - a).approx_equal(b));
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

TYPED_TEST(MatrixTester, trace)
{
    static constexpr std::size_t Rows = TypeParam::value / 100;
    static constexpr std::size_t Cols = TypeParam::value % 100;

    mt::Matrix<Rows, Cols> m(12.3);
    size_t n = std::min(Rows, Cols);
    EXPECT_FLOAT_EQ(n * 12.3, m.trace());

    mt::Matrix<Rows, Cols> i = mt::identity<Rows, Cols>();
    EXPECT_FLOAT_EQ(n, i.trace());
}

TYPED_TEST(MatrixTester, mat_mult)
{
    static constexpr std::size_t Rows = TypeParam::value / 100;
    static constexpr std::size_t Cols = TypeParam::value % 100;

    mt::Matrix<Rows, Cols> a(0.1);
    mt::Matrix<Cols, Rows> b(0.2);

    EXPECT_EQ((a*b).rows(), Rows);
    EXPECT_EQ((a*b).cols(), Rows);
}


//===================================================================//

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}