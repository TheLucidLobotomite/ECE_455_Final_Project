#include <iostream>
#include "numerical_integrator.h"
#include <cmath>
#include <functional> 
#include <omp.h>
#include <iostream>

#include <chrono>

double func(double x) { return sin(x); }

double func2(double x) { return 1/(1-x); }

double func3(double x) { return x*pow(M_E,-x); }


int main() {
    long h=100000000;
    using namespace numint;
   /*
    std::cout << "number of steps " << h << std::endl;
    using namespace numint;
    IntegrationResult result = sequential_integrate(func, 0, M_PI, h);
    std::cout << "Sequential Integral of sin(x) from 0 to pi = " << result.value << std::endl;
    std::cout << "took: " << result.time_s << " sec"<<std::endl;
    IntegrationResult result2= parallel_integrate(func, 0, M_PI, h);
    std::cout << "Parallel Integral of sin(x) from 0 to pi = " << result2.value << std::endl;
    std::cout << "took: " << result2.time_s << " sec"<<std::endl;
    double expected = -cos(M_PI)+cos(0);
    std::cout << "expected result: " << expected << std::endl;
    std::cout << "OMP threads = " << omp_get_max_threads() << "\n";



    
    
    result = sequential_integrate(func3, 2, 12, h);
    std::cout << "Sequential Integral of xe^-x from 2 to 12 = " << result.value << std::endl;
    std::cout << "took: " << result.time_s << " sec"<<std::endl;
    result2= parallel_integrate(func3, 2, 12, h);
    std::cout << "Parallel Integral of xe^-x from 2 to 12 = " << result2.value << std::endl;
    std::cout << "took: " << result2.time_s << " sec"<<std::endl;
     //expected = ((-M_PI-1)/pow(M_E,M_PI))+1;
    std::cout << "expected result: " << 0.40592  << std::endl;

    */ 
   for (int i=3; i<10; i++){
    h=pow(10,i);
   std::cout << "number of steps " << h << std::endl;
    IntegrationResult result = cpv_integrate_s(func2, -8, 10, h,1);
    //result= parallel_integrate(func2, 0, 0.99999, h);
    //IntegrationResult result4=parallel_integrate(func2, 1.00001,  M_PI, h);
    std::cout << "Sequential Integral of 1/(x-1) from -8 to 10 = " << result.value << std::endl;//+result4.value
    std::cout << "took: " << result.time_s << " sec"<<std::endl;
    //result2= parallel_integrate(func2, 0, 0.99999, h);
    IntegrationResult result2= cpv_integrate_p(func2, -8, 10, h,1);
    //IntegrationResult result3=parallel_integrate(func2, 1.00001,  M_PI, h);
    std::cout << "Parallel Integral of 1/(x-1) from -8 to 10 = " << result2.value << std::endl;//+result3.value
    std::cout << "took: " << result2.time_s << " sec"<<std::endl;
   // double expected = std::logb(x);
    std::cout << "expected result: 0" << std::endl;
    }
}
//g++ test.cpp -o test
