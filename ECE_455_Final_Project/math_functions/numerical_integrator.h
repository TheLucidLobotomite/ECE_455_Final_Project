#ifndef NUMERICAL_INTEGRATOR_H
#define NUMERICAL_INTEGRATOR_H

namespace numint {
    struct IntegrationResult {
    double value;   // Computed integral
    double time_s;  // Time in seconds
};

IntegrationResult sequential_integrate(double (*kernel)(double rp, double r), double lower, double upper, int steps, double r);
IntegrationResult parallel_integrate(double (*kernel)(double rp, double r), double lower, double upper, int steps, double r);

IntegrationResult squared_sum(double (*kernel)(double rp, double r), int lower, int upper, double r);
IntegrationResult parallel_squared_sum(double (*kernel)(double rp, double r), int lower, int upper, double r);
IntegrationResult cpv_integrate_p(double (*kernel)(double rp, double r), double lower, double upper, int steps, double singular);
IntegrationResult cpv_integrate_s(double (*kernel)(double rp, double r), double lower, double upper, int steps, double singular);
}


#endif