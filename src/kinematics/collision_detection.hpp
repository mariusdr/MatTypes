#ifndef LIBCUMANIP_KINEMATICS_COLLISION_DETECTION_HPP
#define LIBCUMANIP_KINEMATICS_COLLISION_DETECTION_HPP

#include "../math_types.hpp"
#include "forward_kinematics.hpp"


namespace cumanip
{

struct Sphere
{
    __host__ __device__
    explicit Sphere(): center(mt::vec3f(0.f, 0.f, 0.f)), radius(1.f)
    {}

    __host__ __device__
    Sphere(mt::Vector3f center, float radius): center(center), radius(radius)
    {}

    mt::Vector3f center;
    float radius;
};

__host__ __device__ inline
bool cuttest(const Sphere& x, const Sphere& y)
{
    float center_dist = mt::distance(x.center, y.center); 
    float rad_sum = x.radius + y.radius;

    return (center_dist <= rad_sum);
}

struct Cube 
{
    __host__ __device__
    explicit Cube(): pose(mt::identity<4, 4>()), extents(mt::vec3f(1.f, 1.f, 1.f))
    {}

    __host__ __device__
    Cube(mt::Matrix4f pose, mt::Vector3f extents): pose(pose), extents(extents)
    {}

    __host__ __device__ 
    Cube(mt::Matrix4f pose, float ex, float ey, float ez): pose(pose), extents(mt::vec3f(ex, ey, ez))
    {}

    __host__ __device__
    Cube(mt::Vector3f pos, mt::Matrix3f rot, mt::Vector3f extents): Cube(mt::affine(rot, pos), extents)
    {}

    __host__ __device__
    Cube(mt::Vector3f pos, mt::Matrix3f rot, float ex, float ey, float ez): Cube(mt::affine(rot, pos), mt::vec3f(ex, ey, ez))
    {}

    __host__ __device__ 
    Cube(mt::Vector3f pos, mt::Vector3f rpy, mt::Vector3f extents): Cube(mt::affine(mt::fromRPY(rpy), pos), extents)
    {}
    
    __host__ __device__ 
    Cube(mt::Vector3f pos, mt::Vector3f rpy, float ex, float ey, float ez): Cube(mt::affine(mt::fromRPY(rpy), pos), mt::vec3f(ex, ey, ez))
    {}

    __host__ __device__ 
    mt::Vector3f position() const 
    {
        mt::Matrix3f rot;
        mt::Vector3f pos;
        mt::from_affine(pose, rot, pos);
        return pos;
    }

    __host__ __device__ 
    mt::Matrix3f rotation() const 
    {
        mt::Matrix3f rot;
        mt::Vector3f pos;
        mt::from_affine(pose, rot, pos);
        return rot;
    }

    __host__ __device__
    float min(int dim) const 
    {
        return position().at(dim) - extents.at(dim);
    }   

    __host__ __device__
    float max(int dim) const 
    {
        return position().at(dim) + extents.at(dim);
    }


    mt::Matrix4f pose;
    mt::Vector3f extents;
};

// https://www.gamasutra.com/view/feature/131790/simple_intersection_tests_for_games.php?print=1
// Arvo's algorithm
__host__ __device__ inline 
bool cuttest(const Cube& c, const Sphere& s)
{
    mt::Vector3f center = s.center;

    float d = 0.f;

    for (int i = 0; i < 3; ++i)
    {
        if (center.at(i) < c.min(i))
        {
            float z = center.at(i) - c.min(i); 
            d += z * z;
        }
        else if (center.at(i) > c.max(i))
        {
            float z = center.at(i) - c.max(i);
            d += z * z;
        }
    }

    return d <= s.radius * s.radius;
}

__host__ __device__ inline 
bool cuttest(const Sphere& s, const Cube& c)
{
    return cuttest(c, s);
}

struct RobotVolumeModel
{
    static const int NumSpheres = 14;

    __host__ __device__ 
    RobotVolumeModel()
    {
        // the base sphere should be a bit larger
        spheres[0].radius = 0.0875;

        // bigger segment
        spheres[1].radius = 0.0665;
        spheres[2].radius = 0.0625;
        spheres[3].radius = 0.0625;
        spheres[4].radius = 0.0625;
        spheres[5].radius = 0.0675;

        // smaller segment
        spheres[6].radius = 0.05;
        spheres[7].radius = 0.05;
        spheres[8].radius = 0.05;
        spheres[9].radius = 0.05;
        spheres[10].radius = 0.05;

        // wrist joints
        spheres[11].radius = 0.065;
        spheres[12].radius = 0.065;

        // tcp sphere 
        spheres[13].radius = 0.025;
    }

    __host__ __device__ 
    void setFromFK(const mt::State& state)
    {
        URKForwardKinematics fk;
        fk.solve(state);

        const mt::Vector3f j1 = mt::translation(fk.get_inverse_base_transform() * fk.transform_base_to_j1());
        const mt::Vector3f j2 = mt::translation(fk.get_inverse_base_transform() * fk.transform_base_to_j2());
        const mt::Vector3f j3 = mt::translation(fk.get_inverse_base_transform() * fk.transform_base_to_j3());
        const mt::Vector3f j4 = mt::translation(fk.get_inverse_base_transform() * fk.transform_base_to_j4());
        const mt::Vector3f j5 = mt::translation(fk.get_inverse_base_transform() * fk.transform_base_to_j5());
        const mt::Vector3f j6 = mt::translation(fk.get_inverse_base_transform() * fk.transform_base_to_j6());
        
        // base sphere, moved downwads so the socket is also covered
        spheres[0].center = j1 + mt::rotation(fk.get_base_transform()) * mt::vec3f(0, 0, -0.0425); 

        // first segment 
        const mt::Vector3f j1m = j1 + mt::rotation(fk.transform_base_to_j1()) * mt::vec3f(0, 0, -0.1325); 
        const mt::Vector3f j2m = j2 + mt::rotation(fk.transform_base_to_j2()) * mt::vec3f(0, 0, -0.1325);

        spheres[1].center = j1m; 
        spheres[2].center = 0.75 * j1m + 0.25 * j2m;
        spheres[3].center = 0.50 * j1m + 0.50 * j2m;
        spheres[4].center = 0.25 * j1m + 0.75 * j2m;
        spheres[5].center = j2m;

        // second segment
        const mt::Vector3f j2f = j2 + mt::rotation(fk.transform_base_to_j2()) * mt::vec3f(0, 0, -0.015);
        const mt::Vector3f j3f = j3 + mt::rotation(fk.transform_base_to_j3()) * mt::vec3f(0, 0, -0.015);

        spheres[6].center = j2f;
        spheres[7].center = 0.75 * j2f + 0.25 * j3f;
        spheres[8].center = 0.50 * j2f + 0.50 * j3f;
        spheres[9].center = 0.25 * j2f + 0.75 * j3f;
        spheres[10].center = j3f;

        // wrist joints
        spheres[11].center = j4;
        spheres[12].center = j5;
        spheres[13].center = j6;
    }

    __host__ __device__ 
    bool in_self_collision() const 
    {
        bool cut = false;

        for (int i = 0; i < NumSpheres; ++i)
        {
            for (int j = i + 2; j < NumSpheres; ++j)
            {
                cut |= cuttest(spheres[i], spheres[j]);
            }
        } 
        return cut;
    }

    __host__ __device__ 
    int num_spheres() const
    {
        return NumSpheres;
    }

    Sphere spheres[NumSpheres];
};

template<size_t MaxNumModels>
struct EnvironmentModel
{
    __host__ __device__ 
    EnvironmentModel(): num_models(0) 
    {}
   
    __host__ __device__ 
    void add_model(Cube c)
    {
        models[num_models++] = c;
    }

    __host__ __device__ 
    bool in_collision_with_robot(RobotVolumeModel& robot)
    {
        size_t NS = RobotVolumeModel::NumSpheres;
        bool cut = false;
        for (size_t k = 0; k < num_models; ++k)
        {
            Cube c = models[k];

            for (size_t l = 0; l < NS; ++l)
            {
                Sphere s = robot.spheres[l];
                cut |= cuttest(c, s);

                std::cout << "cut ? " << cut << "\n";
            }
        }
        return cut;
    }

    size_t num_models;
    Cube models[MaxNumModels];
};




} // namespace
#endif