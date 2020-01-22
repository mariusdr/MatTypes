#include <iostream>

#include "types.hpp"

using namespace math_types;

int main()
{
    // Vector3f x = vec3f(1,0,0);

    // std::cout << (x *1.23).length() << "\n";

    // x = vec3f(23,21,11);
    // std::cout << x.length() << "\n";
    // std::cout << x.normalized().length() << "\n";
    // std::cout << rad_to_deg(angle(vec3f(1,0,0), vec3f(0,1,0))) << "\n";

    // std::cout << (x * 5 != 5 * x) << "\n";

    Matrix4f x = mat4f_rows(
        vec4f(1,2,3,4),
        vec4f(5,6,7,8),
        vec4f(9,10,11,12),
        vec4f(13,14,15,16)
    );

    Matrix3f y = mat3f_rows(
        vec3f(0,1,2),
        vec3f(3,4,5),
        vec3f(6,7,8)
    );

    Matrix<3, 4> z = mat3x4_rows(
        vec4f(0,1,2,3),
        vec4f(4,5,6,7),
        vec4f(8,9,10,11)
    );

    Matrix<3, 4> m;
    // m.set_col(0, vec3f(0,1,2));

    m.set_row(1, vec4f(0,1,2,3));
    std::cout << m << "\n";

    std::cout << z << std::endl;
    std::cout << "\n";
    std::cout << z.transpose() << std::endl;

    return EXIT_SUCCESS;
}