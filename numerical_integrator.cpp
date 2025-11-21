#include <functional> 
#include <omp.h>
#include <iostream>
#include "numerical_integrator.h"
#include <cmath>
#include <chrono>

// to run porgrams with this library use: g++ -fopenmp my_program.cpp math_functions/numerical_integrator.cpp -o my_program
namespace numint {
// sequential numerical integration by trapazoidal reimann sum: ((b-a/)2n)*sum((f(a)+2f(a+(b-a)/n))+2f(a+2(b-a)/n))+.....+f(b))
IntegrationResult sequential_integrate(double (*func)(double), double lower, double upper, int steps) { // lower= lower bound, upper=upper bound, step=number of steps, func= function to be integrated over
using namespace std::chrono;
auto start = high_resolution_clock::now(); //start clock for runtime
double step_size =(upper-lower)/steps; // calculate step size
double sum = 0;
for(int i=1; i<steps; i++){  
    double x = lower + i * step_size;
    sum+=func(x);   // sum function values for each step size over the integral per the trapazoidal reimann sum equation 
}
auto end = high_resolution_clock::now();
duration<double> elapsed = end - start;// end clock and find runtime
sum += 0.5 * (func(lower) + func(upper));
return {step_size*sum, elapsed.count()};

}
// parrallelized numerical integration by trapazoidal reimann sum
IntegrationResult parallel_integrate(double (*func)(double), double lower, double upper, int steps) { // lower= lower bound, upper=upper bound, step=number of steps, func= function to be integrated over
using namespace std::chrono;   

auto start = high_resolution_clock::now();    //start clock for runtime
double step_size =(upper-lower)/steps; // calculate step size
double sum = 0;
std::cout << "OMP threads = " << omp_get_max_threads() << "\n";
omp_set_num_threads(4);
#pragma omp parallel for reduction(+:sum)   // parallelize the for loop
for(int i=1; i<steps; i++){ 
double x = lower + i * step_size;
sum+=func(x);  // sum function values for each step size over the integral per the trapazoidal reimann sum equation 
}
    

sum += 0.5 * (func(lower) + func(upper));
auto end = high_resolution_clock::now();
duration<double> elapsed = end - start; // end clock and find runtime

return {step_size*sum, elapsed.count()}; // multiply by step size  and return with runtime
    

}

IntegrationResult squared_sum(std::vector<double>& vec){
    using namespace std::chrono;
    auto start = high_resolution_clock::now();  //start clock for runtime
    double sum=0;
    for(int i=0; i<=sizeof(vec); i++){  // sum fucntion over bounds
        sum+=pow(abs(vec[i]),2);
    }


auto end = high_resolution_clock::now();
duration<double> elapsed = end - start; // end clock and find runtime

return {sum, elapsed.count()}; // return sum and runtime
}

IntegrationResult parallel_squared_sum(double func[]){
    using namespace std::chrono;
    auto start = high_resolution_clock::now();  //start clock for runtime
    double sum=0;
   #pragma omp parallel for reduction(+:sum)     // parallelize the for loop
    double sum=0;
    for(int i=0; i<=sizeof(func); i++){  // sum fucntion over bounds
        sum+=pow(abs(func[i]),2);
    }


auto end = high_resolution_clock::now();// end clock and find runtime
duration<double> elapsed = end - start;

return {sum, elapsed.count()}; // return sum and runtime
}


IntegrationResult cpv_integrate_s(double (*func)(double),double lower, double upper, int steps, double singular)
{
    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    //const double singular = 1.0;  // location of vertical asymptote
    const double eps = 1e-8;      // exclusion size near singularity

    if (!(lower < singular && upper > singular)) {
        // no singularity inside interval → normal integration works
        return sequential_integrate(func, lower, upper, steps);
    }

    // Sub-intervals: [lower, 1−ε] and [1+ε, upper]
    double a1 = lower, b1 = singular - eps;
    double a2 = singular + eps, b2 = upper;

    double I1 = sequential_integrate(func, a1, b1, steps).value;
    double I2 = sequential_integrate(func, a2, b2, steps).value;

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;

    return { I1 + I2, elapsed.count() };
}

IntegrationResult cpv_integrate_p(double (*func)(double),double lower, double upper, int steps, double singular)
{
    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    //const double singular = 1.0;  // location of vertical asymptote
    const double eps = 1e-8;      // exclusion size near singularity

    if (!(lower < singular && upper > singular)) {
        // no singularity inside interval → normal integration works
        return parallel_integrate(func, lower, upper, steps);
    }

    // Sub-intervals: [lower, 1−ε] and [1+ε, upper]
    double a1 = lower, b1 = singular - eps;
    double a2 = singular + eps, b2 = upper;

    double I1 = parallel_integrate(func, a1, b1, steps).value;
    double I2 = parallel_integrate(func, a2, b2, steps).value;

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;

    return { I1 + I2, elapsed.count() };
}


}