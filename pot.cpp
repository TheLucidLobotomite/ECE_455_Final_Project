#include <iostream>
#include "numerical_integrator.h"
#include <cmath>
#include <functional> 
#include <omp.h>
#include <iostream>

#include <chrono>

namespace numint {

double func(double x) { return 1/(1-x); }



IntegrationResult Hartree_potential_s(std::vector<double>& wavefunctions, double radius, int steps, int i){
    using namespace std::chrono;
    auto start = high_resolution_clock::now();  //start clock for runtime
    //function call 
    double sum=0;
    for(int i=0; i<=sizeof(wavefunctions); i++){  // sum fucntion over bounds
        sum+=pow(abs(wavefunctions[i]),2);
    }
    IntegrationResult fun = sequential_integrate(func,0,radius,steps);
    double result=fun.value*sum*4*M_PI*1.60217663*1.60217663*pow(10,-38);


auto end = high_resolution_clock::now();
duration<double> elapsed = end - start; // end clock and find runtime

return {result, elapsed.count()}; // return sum and runtime
}

IntegrationResult Hartree_potential_p(std::vector<double>& wavefunctions, double radius, int steps, int i){
    using namespace std::chrono;
    auto start = high_resolution_clock::now();  //start clock for runtime
     double sum=0;
     
    for(int i=0; i<=sizeof(wavefunctions); i++){  // sum fucntion over bounds
        sum+=pow(abs(wavefunctions[i]),2);
    }
    IntegrationResult fun = sequential_integrate(func,0,radius,steps);
    double result=fun.value*sum*4*M_PI*1.60217663*1.60217663*pow(10,-38);


  
  


auto end = high_resolution_clock::now();
duration<double> elapsed = end - start; // end clock and find runtime

return {sum, elapsed.count()}; // return sum and runtime
}
}