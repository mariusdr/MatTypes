#ifndef LIBCUMANIP_KINEMATICS_CUDA_UR_KINEMATICS_HPP
#define LIBCUMANIP_KINEMATICS_CUDA_UR_KINEMATICS_HPP

#include "../math_types.hpp"

#ifndef UR5_PARAMS
#define UR5_PARAMS
#endif
//#include <inrop/libcumanip/DHParams.hpp>
#ifdef UR5_PARAMS
const double d1 =  0.089159;
const double a2 = -0.42500;
const double a3 = -0.39225;
const double d4 =  0.10915;
const double d5 =  0.09465;
const double d6 =  0.0823;
#endif

#define ZERO_THRESH 0.00000001 // this was from before i changed doubles to floats
#define SIGN(x) ( ( (x) > 0 ) - ( (x) < 0 ) )
#define PI M_PI


namespace cumanip 
{
namespace ur_kin
{

__device__ __host__ inline 
void forward(const float *q, float *T)
{
    const float s1 = sin(q[0]);
    const float c1 = cos(q[0]);
    const float s2 = sin(q[1]);
    const float c2 = cos(q[1]);
    const float s3 = sin(q[2]);
    const float c3 = cos(q[2]);
    const float s5 = sin(q[4]);
    const float c5 = cos(q[4]);
    const float s6 = sin(q[5]);
    const float c6 = cos(q[5]);
    const float s234 = sin(q[1] + q[2] + q[3]);
    const float c234 = cos(q[1] + q[2] + q[3]);

    T[0] = ((c1 * c234 - s1 * s234) * s5) / 2.0 - c5 * s1 + ((c1 * c234 + s1 * s234) * s5) / 2.0;

    T[1] = (c6 * (s1 * s5 + ((c1 * c234 - s1 * s234) * c5) / 2.0 + ((c1 * c234 + s1 * s234) * c5) / 2.0) -
            (s6 * ((s1 * c234 + c1 * s234) - (s1 * c234 - c1 * s234))) / 2.0);

    T[2] = (-(c6 * ((s1 * c234 + c1 * s234) - (s1 * c234 - c1 * s234))) / 2.0 -
            s6 * (s1 * s5 + ((c1 * c234 - s1 * s234) * c5) / 2.0 + ((c1 * c234 + s1 * s234) * c5) / 2.0));

    T[3] = ((d5 * (s1 * c234 - c1 * s234)) / 2.0 - (d5 * (s1 * c234 + c1 * s234)) / 2.0 -
            d4 * s1 + (d6 * (c1 * c234 - s1 * s234) * s5) / 2.0 + (d6 * (c1 * c234 + s1 * s234) * s5) / 2.0 -
            a2 * c1 * c2 - d6 * c5 * s1 - a3 * c1 * c2 * c3 + a3 * c1 * s2 * s3);

    T[4] = c1 * c5 + ((s1 * c234 + c1 * s234) * s5) / 2.0 + ((s1 * c234 - c1 * s234) * s5) / 2.0;

    T[5] = (c6 * (((s1 * c234 + c1 * s234) * c5) / 2.0 - c1 * s5 + ((s1 * c234 - c1 * s234) * c5) / 2.0) +
            s6 * ((c1 * c234 - s1 * s234) / 2.0 - (c1 * c234 + s1 * s234) / 2.0));

    T[6] = (c6 * ((c1 * c234 - s1 * s234) / 2.0 - (c1 * c234 + s1 * s234) / 2.0) -
            s6 * (((s1 * c234 + c1 * s234) * c5) / 2.0 - c1 * s5 + ((s1 * c234 - c1 * s234) * c5) / 2.0));

    T[7] = ((d5 * (c1 * c234 - s1 * s234)) / 2.0 - (d5 * (c1 * c234 + s1 * s234)) / 2.0 + d4 * c1 +
            (d6 * (s1 * c234 + c1 * s234) * s5) / 2.0 + (d6 * (s1 * c234 - c1 * s234) * s5) / 2.0 + d6 * c1 * c5 -
            a2 * c2 * s1 - a3 * c2 * c3 * s1 + a3 * s1 * s2 * s3);

    T[8] = ((c234 * c5 - s234 * s5) / 2.0 - (c234 * c5 + s234 * s5) / 2.0);

    T[9] = ((s234 * c6 - c234 * s6) / 2.0 - (s234 * c6 + c234 * s6) / 2.0 - s234 * c5 * c6);

    T[10] = (s234 * c5 * s6 - (c234 * c6 + s234 * s6) / 2.0 - (c234 * c6 - s234 * s6) / 2.0);

    T[11] = (d1 + (d6 * (c234 * c5 - s234 * s5)) / 2.0 + a3 * (s2 * c3 + c2 * s3) + a2 * s2 -
             (d6 * (c234 * c5 + s234 * s5)) / 2.0 - d5 * c234);
    
    T[12] = 0.f;
    T[13] = 0.f;
    T[14] = 0.f;
    T[15] = 1.f;
}

__device__ __host__ inline
float wrap_angle(float angle)
{
    float mult = 1.f;
    float add = 0.f;

    if (fabs(angle) < ZERO_THRESH)
    {
        mult = 0.f;
    }

    if (angle < 0.f)
    {
        add = 2 * M_PI;
    }

    return mult * (angle + add);
}

__device__ __host__ inline
float div__(float a, float b)
{
    const float diff = fabs(fabs(a) - fabs(b));
    if (diff < ZERO_THRESH)
    {
        a = float(SIGN(a));
        b = float(SIGN(b));
    }
    else 
    {
        a = a / b;
        b = 1.f;
    }
    return a * b;
}

__device__ __host__ inline 
float assign__(float a)
{
    if (fabs(a) < ZERO_THRESH)
        return 0.f;
    return a;
}

__device__ __host__ inline 
bool solve_for_q1(const float* T, float* q1)
{
    const float T02 = - T[0 * 4 + 0];
    const float T03 = -T[0 * 4 + 3];
    const float T12 = -T[1 * 4 + 0];
    const float T13 = -T[1 * 4 + 3];

    const float A = d6 * T12 - T13;
    const float B = d6 * T02 - T03;
    const float R = A * A + B * B;

    bool solver_failed = false;

    const float div_az = div__(-d4, B);
    const float arcsin_az = wrap_angle(asin(div_az));

    const float div_bz = div__(d4, A);
    const float arccos_bz = acos(div_bz);

    const float div = d4 / sqrt(R);
    const float arccos = acos(div);
    const float arctan = atan2(-B, A);

    float pos = arccos + arctan;
    float neg = -arccos + arctan;
    if (fabs(pos) < ZERO_THRESH)
        pos = 0.0;
    if (fabs(neg) < ZERO_THRESH)
        neg = 0.0;

    if (fabs(A) < ZERO_THRESH)
    {
        solver_failed |= (fabs(div_az) > 1.0);
        // std::cout << "div_az : " << div_az << "\n";

        q1[0] = arcsin_az;
        q1[1] = PI - arcsin_az;
    }
    else if (fabs(B) < ZERO_THRESH)
    {
        solver_failed |= (fabs(div_bz) > 1.0);
        
        // std::cout << "div_bz : " << div_bz << "\n";
        
        q1[0] = arccos_bz;
        q1[1] = 2.0 * PI - arccos_bz;
    }
    else
    {
        solver_failed |= (d4 * d4 > R);
        solver_failed |= (fabs(div) > 1.0);

        // std::cout << "d4^2, R : " << d4*d4 << "; " << R << "\n";
        // std::cout << "div : " << div << "\n";

        // std::cout << pos << ", " << neg << "\n";
        if (pos >= 0.0)
            q1[0] = pos;
        else
            q1[0] = 2.0 * PI + pos;
        if (neg >= 0.0)
            q1[1] = neg;
        else
            q1[1] = 2.0 * PI + neg;
    }

    return solver_failed;
}

__device__ __host__ inline
bool solve_for_q5(const float* T, const float* q1, float* q5)
{
    const float T03 = -T[0 * 4 + 3];
    const float T13 = -T[1 * 4 + 3];

    bool solver_failed = false;

    for (int i = 0; i < 2; i++)
    {
        const float numer = (T03 * sin(q1[i]) - T13 * cos(q1[i]) - d4);
        float div; 
        if(fabs(fabs(numer) - fabs(d6)) < ZERO_THRESH)
          div = SIGN(numer) * SIGN(d6);
        else
          div = numer / d6;
        solver_failed |= (fabs(div) > 1.f);

        const float arccos = acos(div);
        q5[i * 2 + 0] = arccos;                  // wrist up
        q5[i * 2 + 1] = 2.0 * PI - arccos;       // wrist down
    }

    return solver_failed;
}

__device__ __host__ inline 
void solve_for_q6(const float *T, float* q6, float q6_des, float s1, float c1, float s5)
{
    const float T00 = T[0 * 4 + 1];
    const float T01 = T[0 * 4 + 2];
    const float T10 = T[1 * 4 + 1];
    const float T11 = T[1 * 4 + 2];

    if (fabs(s5) < ZERO_THRESH)
    {
        *q6 = q6_des;
    }
    else
    {
        *q6 = atan2(SIGN(s5) * -(T01 * s1 - T11 * c1),
                    SIGN(s5) * (T00 * s1 - T10 * c1));
        if (fabs(*q6) < ZERO_THRESH)
            *q6 = 0.0;
        if (*q6 < 0.0)
            *q6 += 2.0 * PI;
    }
}

__host__ __device__ inline 
bool solve_for_q234(const float *T, float *q2, float *q3, float *q4, float s1, float c1, float s5, float c5, float s6, float c6)
{
    const float T02 = - T[0 * 4 + 0];
    const float T00 = T[0 * 4 + 1];
    const float T01 = T[0 * 4 + 2];
    const float T03 = -T[0 * 4 + 3];
    const float T12 = -T[1 * 4 + 0];
    const float T10 = T[1 * 4 + 1];
    const float T11 = T[1 * 4 + 2];
    const float T13 = -T[1 * 4 + 3];
    const float T22 = T[2 * 4 + 0];
    const float T20 = -T[2 * 4 + 1];
    const float T21 = -T[2 * 4 + 2];
    const float T23 = T[2 * 4 + 3];
    
    const float x04x = -s5 * (T02 * c1 + T12 * s1) - c5 * (s6 * (T01 * c1 + T11 * s1) - c6 * (T00 * c1 + T10 * s1));
    const float x04y = c5 * (T20 * c6 - T21 * s6) - T22 * s5;
    const float p13x = d5 * (s6 * (T00 * c1 + T10 * s1) + c6 * (T01 * c1 + T11 * s1)) - d6 * (T02 * c1 + T12 * s1) +
                       T03 * c1 + T13 * s1;
    const float p13y = T23 - d1 - d6 * T22 + d5 * (T21 * c6 + T20 * s6);

    float c3 = (p13x * p13x + p13y * p13y - a2 * a2 - a3 * a3) / (2.0 * a2 * a3);
    if (fabs(fabs(c3) - 1.0) < ZERO_THRESH)
        c3 = float(SIGN(c3));
    else if (fabs(c3) > 1.0)
    {
        return true;
    }
    float arccos = acos(c3);
    q3[0] = arccos;            // elbow up
    q3[1] = 2.0 * PI - arccos; // elbow down
    float denom = a2 * a2 + a3 * a3 + 2 * a2 * a3 * c3;
    if (fabs(denom) < ZERO_THRESH)
    {
        return true;
    }

    const float s3 = sin(arccos);
    const float A = (a2 + a3 * c3), B = a3 * s3;
    q2[0] = atan2((A * p13y - B * p13x) / denom, (A * p13x + B * p13y) / denom);
    q2[1] = atan2((A * p13y + B * p13x) / denom, (A * p13x - B * p13y) / denom);
    float c23_0 = cos(q2[0] + q3[0]);
    float s23_0 = sin(q2[0] + q3[0]);
    float c23_1 = cos(q2[1] + q3[1]);
    float s23_1 = sin(q2[1] + q3[1]);
    q4[0] = atan2(c23_0 * x04y - s23_0 * x04x, x04x * c23_0 + x04y * s23_0);
    q4[1] = atan2(c23_1 * x04y - s23_1 * x04x, x04x * c23_1 + x04y * s23_1);

    for (int k = 0; k < 2; ++k)
    {
        if (fabs(q2[k]) < ZERO_THRESH)
            q2[k] = 0.0;
        else if (q2[k] < 0.0)
            q2[k] += 2.0 * PI;
        if (fabs(q4[k]) < ZERO_THRESH)
            q4[k] = 0.0;
        else if (q4[k] < 0.0)
            q4[k] += 2.0 * PI;
    }
    return false;
}

__device__ __host__ inline
int backward(const float *T, float *q_sols, unsigned char* status, float q6_des)
{
    bool solver_failed = false;

    float q1[2]; 
    float q5[2][2];
    float q6[2][2];
    float q2[2][2][2];
    float q3[2][2][2];
    float q4[2][2][2];
    
    ////////////////////////////// shoulder rotate joint (q1) //////////////////////////////

    bool q1_failed = solve_for_q1(T, q1);
    solver_failed |= q1_failed;

    //q1[0] -> shoulder left
    //q2[1] -> shoulder right

    ////////////////////////////// wrist 2 joint (q5) //////////////////////////////
    
    bool q5_failed = solve_for_q5(T, q1, &(q5[0][0]));
    solver_failed |= q5_failed;
    
    // q5[i][0] -> wrist up
    // q5[i][1] -> wrist down

    ////////////////////////////////////////////////////////////////////////////////

    int num_sols = 0;

    bool q234_failed = false;

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            const float c1 = cos(q1[i]);
            const float s1 = sin(q1[i]);
            const float c5 = cos(q5[i][j]);
            const float s5 = sin(q5[i][j]);

            ////////////////////////////// wrist 3 joint (q6) //////////////////////////////

            solve_for_q6(T, &(q6[i][j]), q6_des, s1, c1, s5);

            ////////////////////////////////////////////////////////////////////////////////

            const float c6 = cos(q6[i][j]);
            const float s6 = sin(q6[i][j]);
            
            ///////////////////////////// RRR joints (q2,q3,q4) ////////////////////////////

            q234_failed = solve_for_q234(T, q2[i][j], q3[i][j], q4[i][j], s1, c1, s5, c5, s6, c6);

            // q3[i][j][0] -> elbow up
            // q3[i][j][1] -> elbow down
            
            ////////////////////////////////////////////////////////////////////////////////

            if (!solver_failed && !q234_failed)
            {
                for (int k = 0; k < 2; ++k)
                {
                    q_sols[num_sols * 6 + 0] = q1[i];
                    q_sols[num_sols * 6 + 1] = q2[i][j][k];
                    q_sols[num_sols * 6 + 2] = q3[i][j][k];
                    q_sols[num_sols * 6 + 3] = q4[i][j][k];
                    q_sols[num_sols * 6 + 4] = q5[i][j];
                    q_sols[num_sols * 6 + 5] = q6[i][j];

                    status[num_sols] |= i << 2;
                    status[num_sols] |= j << 1;
                    status[num_sols] |= k;

                    num_sols++;
                }
            }
        }
    }
    return num_sols;
}

__device__ __host__ inline
void forward_all(const float *q, float *T1, float *T2, float *T3, float *T4, float *T5, float *T6)
{
    float s1 = sin(*q), c1 = cos(*q);
    q++; // q1
    float q23 = *q, q234 = *q, s2 = sin(*q), c2 = cos(*q);
    q++; // q2
    float s3 = sin(*q), c3 = cos(*q);
    q23 += *q;
    q234 += *q;
    q++; // q3
    q234 += *q;
    q++; // q4
    float s5 = sin(*q), c5 = cos(*q);
    q++;                               // q5
    float s6 = sin(*q), c6 = cos(*q); // q6
    float s23 = sin(q23), c23 = cos(q23);
    float s234 = sin(q234), c234 = cos(q234);

    if (T1 != NULL)
    {
        *T1 = c1;
        T1++;
        *T1 = 0;
        T1++;
        *T1 = s1;
        T1++;
        *T1 = 0;
        T1++;
        *T1 = s1;
        T1++;
        *T1 = 0;
        T1++;
        *T1 = -c1;
        T1++;
        *T1 = 0;
        T1++;
        *T1 = 0;
        T1++;
        *T1 = 1;
        T1++;
        *T1 = 0;
        T1++;
        *T1 = d1;
        T1++;
        *T1 = 0;
        T1++;
        *T1 = 0;
        T1++;
        *T1 = 0;
        T1++;
        *T1 = 1;
        T1++;
    }

    if (T2 != NULL)
    {
        *T2 = c1 * c2;
        T2++;
        *T2 = -c1 * s2;
        T2++;
        *T2 = s1;
        T2++;
        *T2 = a2 * c1 * c2;
        T2++;
        *T2 = c2 * s1;
        T2++;
        *T2 = -s1 * s2;
        T2++;
        *T2 = -c1;
        T2++;
        *T2 = a2 * c2 * s1;
        T2++;
        *T2 = s2;
        T2++;
        *T2 = c2;
        T2++;
        *T2 = 0;
        T2++;
        *T2 = d1 + a2 * s2;
        T2++;
        *T2 = 0;
        T2++;
        *T2 = 0;
        T2++;
        *T2 = 0;
        T2++;
        *T2 = 1;
        T2++;
    }

    if (T3 != NULL)
    {
        *T3 = c23 * c1;
        T3++;
        *T3 = -s23 * c1;
        T3++;
        *T3 = s1;
        T3++;
        *T3 = c1 * (a3 * c23 + a2 * c2);
        T3++;
        *T3 = c23 * s1;
        T3++;
        *T3 = -s23 * s1;
        T3++;
        *T3 = -c1;
        T3++;
        *T3 = s1 * (a3 * c23 + a2 * c2);
        T3++;
        *T3 = s23;
        T3++;
        *T3 = c23;
        T3++;
        *T3 = 0;
        T3++;
        *T3 = d1 + a3 * s23 + a2 * s2;
        T3++;
        *T3 = 0;
        T3++;
        *T3 = 0;
        T3++;
        *T3 = 0;
        T3++;
        *T3 = 1;
        T3++;
    }

    if (T4 != NULL)
    {
        *T4 = c234 * c1;
        T4++;
        *T4 = s1;
        T4++;
        *T4 = s234 * c1;
        T4++;
        *T4 = c1 * (a3 * c23 + a2 * c2) + d4 * s1;
        T4++;
        *T4 = c234 * s1;
        T4++;
        *T4 = -c1;
        T4++;
        *T4 = s234 * s1;
        T4++;
        *T4 = s1 * (a3 * c23 + a2 * c2) - d4 * c1;
        T4++;
        *T4 = s234;
        T4++;
        *T4 = 0;
        T4++;
        *T4 = -c234;
        T4++;
        *T4 = d1 + a3 * s23 + a2 * s2;
        T4++;
        *T4 = 0;
        T4++;
        *T4 = 0;
        T4++;
        *T4 = 0;
        T4++;
        *T4 = 1;
        T4++;
    }

    if (T5 != NULL)
    {
        *T5 = s1 * s5 + c234 * c1 * c5;
        T5++;
        *T5 = -s234 * c1;
        T5++;
        *T5 = c5 * s1 - c234 * c1 * s5;
        T5++;
        *T5 = c1 * (a3 * c23 + a2 * c2) + d4 * s1 + d5 * s234 * c1;
        T5++;
        *T5 = c234 * c5 * s1 - c1 * s5;
        T5++;
        *T5 = -s234 * s1;
        T5++;
        *T5 = -c1 * c5 - c234 * s1 * s5;
        T5++;
        *T5 = s1 * (a3 * c23 + a2 * c2) - d4 * c1 + d5 * s234 * s1;
        T5++;
        *T5 = s234 * c5;
        T5++;
        *T5 = c234;
        T5++;
        *T5 = -s234 * s5;
        T5++;
        *T5 = d1 + a3 * s23 + a2 * s2 - d5 * c234;
        T5++;
        *T5 = 0;
        T5++;
        *T5 = 0;
        T5++;
        *T5 = 0;
        T5++;
        *T5 = 1;
        T5++;
    }

    if (T6 != NULL)
    {
        *T6 = c6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * s6;
        T6++;
        *T6 = -s6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * c6;
        T6++;
        *T6 = c5 * s1 - c234 * c1 * s5;
        T6++;
        *T6 = d6 * (c5 * s1 - c234 * c1 * s5) + c1 * (a3 * c23 + a2 * c2) + d4 * s1 + d5 * s234 * c1;
        T6++;
        *T6 = -c6 * (c1 * s5 - c234 * c5 * s1) - s234 * s1 * s6;
        T6++;
        *T6 = s6 * (c1 * s5 - c234 * c5 * s1) - s234 * c6 * s1;
        T6++;
        *T6 = -c1 * c5 - c234 * s1 * s5;
        T6++;
        *T6 = s1 * (a3 * c23 + a2 * c2) - d4 * c1 - d6 * (c1 * c5 + c234 * s1 * s5) + d5 * s234 * s1;
        T6++;
        *T6 = c234 * s6 + s234 * c5 * c6;
        T6++;
        *T6 = c234 * c6 - s234 * c5 * s6;
        T6++;
        *T6 = -s234 * s5;
        T6++;
        *T6 = d1 + a3 * s23 + a2 * s2 - d5 * c234 - d6 * s234 * s5;
        T6++;
        *T6 = 0;
        T6++;
        *T6 = 0;
        T6++;
        *T6 = 0;
        T6++;
        *T6 = 1;
        T6++;
    }
}


} // namespace ur_kin
} // namespace cumanip
#endif