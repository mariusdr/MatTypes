#include <gtest/gtest.h>

#include "Types.hpp"
#include "kin.h"

#include <cstdlib>

int backward_ref(const float* T, float* q_sols, float q6_des, const ur_kin::DHParams& dh);
void forward_ref(const float* q, float* T, const ur_kin::DHParams& dh);

class KinematicsTester : public ::testing::Test {};

TEST(KinematicsTester, forward_consistent_with_autogen_solver)
{
    srand(time(nullptr));
    for (int i = 0; i < 10000; ++i)
    {
        mt::State s(0.f);
        for (int k = 0; k < 6; ++k)
        {
            float sign = -1.f;
            if (rand() % 2) 
                sign = 1.f;

            float mult = 1.f;
            if (i > 100)
              mult = 10.f;
            if (i > 1000)
              mult = 100.f;

            s.data[k] = 10.f * sign * ((float)rand()/(float)(RAND_MAX/M_2_PI));
        }

        mt::Matrix4f sol(0.f);
        mt::Matrix4f ref(0.f);

        forward_ref(s.data, ref.data, ur_kin::UR_5_DH);
        ur_kin::forward(s.data, sol.data, ur_kin::UR_5_DH);

        float prec = 1e-8;
        if (!ref.approx_equal(sol, prec))
        {
            std::cout << std::fixed;
            std::cout << std::setprecision(6);

            std::cout << "state: " << s << "\n";
            std::cout << "ref: \n" << ref << "\n";
            std::cout << "sol: \n" << sol << "\n";
            std::cout << "delta: \n" << sol-ref << "\n";
        }
        ASSERT_TRUE(ref.approx_equal(sol, prec));
    }
}

TEST(KinematicsTester, backward_consistent_with_autogen_solver)
{
    srand(time(nullptr));
    for (int i = 0; i < 10000; ++i)
    {
        mt::State s(0.f);
        for (int k = 0; k < 6; ++k)
        {
            float sign = -1.f;
            if (rand() % 2) 
                sign = 1.f;

            float mult = 1.f;
            if (i > 100)
              mult = 10.f;
            if (i > 1000)
              mult = 100.f;

            s.data[k] = 10.f * sign * ((float)rand()/(float)(RAND_MAX/M_2_PI));
        }

        mt::Matrix4f fwd = ur_kin::solveFK(s);

        unsigned char status[8];
        mt::Matrix<8, 6> sol(0.f);
        int num_of_sols = ur_kin::backward(fwd.data, sol.data, status, 0, ur_kin::UR_5_DH);

        mt::Matrix<8, 6> ref(0.f);
        int num_of_ref_sols = backward_ref(fwd.data, ref.data, 0.f, ur_kin::UR_5_DH);

        float prec = 1e-8;
        if (!ref.approx_equal(sol, prec))
        {
            std::cout << std::fixed;
            std::cout << std::setprecision(6);

            std::cout << "state: " << s << "\n";
            std::cout << "num of reference solutions: " << num_of_ref_sols << "\n";
            std::cout << "num of solutions: " << num_of_sols << "\n";
            std::cout << "ref: \n" << ref << "\n";
            std::cout << "sol: \n" << sol << "\n";
            std::cout << "delta: \n" << sol-ref << "\n";
        }
        ASSERT_TRUE(ref.approx_equal(sol, prec));
    }
}


TEST(KinematicsTester, forward_and_backward_consistent)
{
    srand(time(nullptr));
    for (int i = 0; i < 10000; ++i)
    {
        mt::State s(0.f);
        for (int k = 0; k < 6; ++k)
        {
            float sign = -1.f;
            if (rand() % 2) 
                sign = 1.f;

            // float mult = 1.f;
            // if (i > 100)
            //   mult = 10.f;
            // if (i > 1000)
            //   mult = 100.f;

            s.data[k] = 10.f * sign * ((float)rand()/(float)(RAND_MAX/M_2_PI));
        }

        mt::Matrix4f fwd = ur_kin::solveFK(s);
        mt::Matrix<8, 6> ik_sol(0.f);
        unsigned char status[8];
        int num_solutions = ur_kin::solveIK(fwd, ik_sol, status, 0.f);

        EXPECT_GT(num_solutions, 0);
        if (num_solutions == 0)
        {
            std::cout << "state: " << s << "\n";
            std::cout << "fwd:\n" << fwd << "\n";
            std::cout << "ik_sol:\n" << ik_sol << "\n";
        }
    }
}

//====================================================================================================================//

int backward_ref(const float* T, float* q_sols, float q6_des, const ur_kin::DHParams& dh)
{
    int num_sols = 0;
    float T02 = -*T; T++; float T00 =  *T; T++; float T01 =  *T; T++; float T03 = -*T; T++; 
    float T12 = -*T; T++; float T10 =  *T; T++; float T11 =  *T; T++; float T13 = -*T; T++; 
    float T22 =  *T; T++; float T20 = -*T; T++; float T21 = -*T; T++; float T23 =  *T;

    const float d1 = dh.d1;
    const float a2 = dh.a2;
    const float a3 = dh.a3;
    const float d4 = dh.d4;
    const float d5 = dh.d5;
    const float d6 = dh.d6;

    ////////////////////////////// shoulder rotate joint (q1) //////////////////////////////
    float q1[2];
    {
      float A = d6*T12 - T13;
      float B = d6*T02 - T03;
      float R = A*A + B*B;
      if(fabs(A) < ZERO_THRESH) {
        float div;
        if(fabs(fabs(d4) - fabs(B)) < ZERO_THRESH)
          div = -SIGN(d4)*SIGN(B);
        else
          div = -d4/B;

        if (fabs(div) > 1.0)
        {
            return num_sols;
        }
        float arcsin = asin(div);
        if(fabs(arcsin) < ZERO_THRESH)
          arcsin = 0.0;
        if(arcsin < 0.0)
          q1[0] = arcsin + 2.0*PI;
        else
          q1[0] = arcsin;
        q1[1] = PI - arcsin;
      }
      else if(fabs(B) < ZERO_THRESH) {
        float div;
        if(fabs(fabs(d4) - fabs(A)) < ZERO_THRESH)
          div = SIGN(d4)*SIGN(A);
        else
        {
          div = d4/A;
        }

        if (fabs(div) > 1.0)
        {
            return num_sols;
        }

        float arccos = acos(div);
        q1[0] = arccos;
        q1[1] = 2.0*PI - arccos;
      }
      else if(d4*d4 > R) {
        return num_sols;
      }
      else {
        float div = d4 / sqrt(R);
        if (fabs(div) > 1.0)
        {
            return num_sols;
        }
        float arccos = acos(div) ;
        float arctan = atan2(-B, A);
        float pos = arccos + arctan;
        float neg = -arccos + arctan;
        if(fabs(pos) < ZERO_THRESH)
          pos = 0.0;
        if(fabs(neg) < ZERO_THRESH)
          neg = 0.0;
        if(pos >= 0.0)
          q1[0] = pos;
        else
          q1[0] = 2.0*PI + pos;
        if(neg >= 0.0)
          q1[1] = neg; 
        else
          q1[1] = 2.0*PI + neg;
      }
    }
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////// wrist 2 joint (q5) //////////////////////////////
    float q5[2][2];
    {
      for(int i=0;i<2;i++) {
        float numer = (T03*sin(q1[i]) - T13*cos(q1[i])-d4);
        float div;
        if(fabs(fabs(numer) - fabs(d6)) < ZERO_THRESH)
          div = SIGN(numer) * SIGN(d6);
        else
          div = numer / d6;

        if (fabs(div) > 1.0)
        {
            return 0;
        }

        float arccos = acos(div);
        q5[i][0] = arccos;
        q5[i][1] = 2.0*PI - arccos;
      }
    }
    ////////////////////////////////////////////////////////////////////////////////

     {
       for(int i=0;i<2;i++) {
         for(int j=0;j<2;j++) {
           float c1 = cos(q1[i]), s1 = sin(q1[i]);
           float c5 = cos(q5[i][j]), s5 = sin(q5[i][j]);
           float q6;
           ////////////////////////////// wrist 3 joint (q6) //////////////////////////////
           if(fabs(s5) < ZERO_THRESH)
             q6 = q6_des;
           else {
             q6 = atan2(SIGN(s5)*-(T01*s1 - T11*c1), 
                        SIGN(s5)*(T00*s1 - T10*c1));
             if(fabs(q6) < ZERO_THRESH)
               q6 = 0.0;
             if(q6 < 0.0)
               q6 += 2.0*PI;
           }
           ////////////////////////////////////////////////////////////////////////////////

           float q2[2], q3[2], q4[2];
           ///////////////////////////// RRR joints (q2,q3,q4) ////////////////////////////
           float c6 = cos(q6), s6 = sin(q6);
           float x04x = -s5*(T02*c1 + T12*s1) - c5*(s6*(T01*c1 + T11*s1) - c6*(T00*c1 + T10*s1));
           float x04y = c5*(T20*c6 - T21*s6) - T22*s5;
           float p13x = d5*(s6*(T00*c1 + T10*s1) + c6*(T01*c1 + T11*s1)) - d6*(T02*c1 + T12*s1) + 
                         T03*c1 + T13*s1;
           float p13y = T23 - d1 - d6*T22 + d5*(T21*c6 + T20*s6);

           float c3 = (p13x*p13x + p13y*p13y - a2*a2 - a3*a3) / (2.0*a2*a3);
           if(fabs(fabs(c3) - 1.0) < ZERO_THRESH)
             c3 = SIGN(c3);
           else if(fabs(c3) > 1.0) {
             // TODO NO SOLUTION
             continue;
           }
           float arccos = acos(c3);
           q3[0] = arccos;
           q3[1] = 2.0*PI - arccos;
           float denom = a2*a2 + a3*a3 + 2*a2*a3*c3;
           if (fabs(denom) < ZERO_THRESH)
           {
               continue;
           }

           float s3 = sin(arccos);
           float A = (a2 + a3*c3), B = a3*s3;
           q2[0] = atan2((A*p13y - B*p13x) / denom, (A*p13x + B*p13y) / denom);
           q2[1] = atan2((A*p13y + B*p13x) / denom, (A*p13x - B*p13y) / denom);
           float c23_0 = cos(q2[0]+q3[0]);
           float s23_0 = sin(q2[0]+q3[0]);
           float c23_1 = cos(q2[1]+q3[1]);
           float s23_1 = sin(q2[1]+q3[1]);
           q4[0] = atan2(c23_0*x04y - s23_0*x04x, x04x*c23_0 + x04y*s23_0);
           q4[1] = atan2(c23_1*x04y - s23_1*x04x, x04x*c23_1 + x04y*s23_1);
           ////////////////////////////////////////////////////////////////////////////////
           for(int k=0;k<2;k++) {
             if(fabs(q2[k]) < ZERO_THRESH)
               q2[k] = 0.0;
             else if(q2[k] < 0.0) q2[k] += 2.0*PI;
             if(fabs(q4[k]) < ZERO_THRESH)
               q4[k] = 0.0;
             else if(q4[k] < 0.0) q4[k] += 2.0*PI;
             q_sols[num_sols*6+0] = q1[i];    q_sols[num_sols*6+1] = q2[k]; 
             q_sols[num_sols*6+2] = q3[k];    q_sols[num_sols*6+3] = q4[k]; 
             q_sols[num_sols*6+4] = q5[i][j]; q_sols[num_sols*6+5] = q6; 
             num_sols++;
           }
       }
     }
     return num_sols;
   }
}

void forward_ref(const float* q, float* T, const ur_kin::DHParams& dh)
{
    float s1 = sin(*q), c1 = cos(*q); q++;
    float q234 = *q, s2 = sin(*q), c2 = cos(*q); q++;
    float s3 = sin(*q), c3 = cos(*q); q234 += *q; q++;
    q234 += *q; q++;
    float s5 = sin(*q), c5 = cos(*q); q++;
    float s6 = sin(*q), c6 = cos(*q); 
    float s234 = sin(q234), c234 = cos(q234);

    const float d1 = dh.d1;
    const float a2 = dh.a2;
    const float a3 = dh.a3;
    const float d4 = dh.d4;
    const float d5 = dh.d5;
    const float d6 = dh.d6;

    *T = ((c1*c234-s1*s234)*s5)/2.0 - c5*s1 + ((c1*c234+s1*s234)*s5)/2.0; T++;
    *T = (c6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0) - 
          (s6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0); T++;
    *T = (-(c6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0 - 
          s6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0)); T++;
    *T = ((d5*(s1*c234-c1*s234))/2.0 - (d5*(s1*c234+c1*s234))/2.0 - 
          d4*s1 + (d6*(c1*c234-s1*s234)*s5)/2.0 + (d6*(c1*c234+s1*s234)*s5)/2.0 - 
          a2*c1*c2 - d6*c5*s1 - a3*c1*c2*c3 + a3*c1*s2*s3); T++;
    *T = c1*c5 + ((s1*c234+c1*s234)*s5)/2.0 + ((s1*c234-c1*s234)*s5)/2.0; T++;
    *T = (c6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0) + 
          s6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0)); T++;
    *T = (c6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0) - 
          s6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0)); T++;
    *T = ((d5*(c1*c234-s1*s234))/2.0 - (d5*(c1*c234+s1*s234))/2.0 + d4*c1 + 
          (d6*(s1*c234+c1*s234)*s5)/2.0 + (d6*(s1*c234-c1*s234)*s5)/2.0 + d6*c1*c5 - 
          a2*c2*s1 - a3*c2*c3*s1 + a3*s1*s2*s3); T++;
    *T = ((c234*c5-s234*s5)/2.0 - (c234*c5+s234*s5)/2.0); T++;
    *T = ((s234*c6-c234*s6)/2.0 - (s234*c6+c234*s6)/2.0 - s234*c5*c6); T++;
    *T = (s234*c5*s6 - (c234*c6+s234*s6)/2.0 - (c234*c6-s234*s6)/2.0); T++;
    *T = (d1 + (d6*(c234*c5-s234*s5))/2.0 + a3*(s2*c3+c2*s3) + a2*s2 - 
         (d6*(c234*c5+s234*s5))/2.0 - d5*c234); T++;
    *T = 0.0; T++; *T = 0.0; T++; *T = 0.0; T++; *T = 1.0;
}


//====================================================================================================================//