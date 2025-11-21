#ifndef NUMERICAL_INTEGRATOR_H
#define NUMERICAL_INTEGRATOR_H

namespace numint {
    struct IntegrationResult {
    double value;   // Computed integral
    double time_s;  // Time in seconds
};

IntegrationResult sequential_integrate(double (*f)(double), double lower, double upper, int steps);
IntegrationResult parallel_integrate(double (*f)(double), double lower, double upper, int steps);

IntegrationResult squared_sum(double (*f)(double), int lower, int upper);
IntegrationResult parallel_squared_sum(double (*f)(double), int lower, int upper);
IntegrationResult cpv_integrate_p(double (*func)(double),double lower, double upper, int steps, double singular);
IntegrationResult cpv_integrate_s(double (*func)(double),double lower, double upper, int steps, double singular);
}


#endif