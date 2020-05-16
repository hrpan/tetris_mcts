#ifndef SPECIAL_H
#define SPECIAL_H
#include<cmath>

double std_quantile(double nu, double t){
    double nu_half = nu / 2;
    double factor = exp(lgamma(nu_half - 0.5) - lgamma(nu_half));

    double nomin = pow(nu, nu_half);
    double denom = 2 * sqrt(M_PI);

    double c = nomin / denom * factor;

    return pow(c * t, 1 / nu);
}

double std_quantile2(double nu, double t){
    double alpha = 2 / t;
    double f_nu = 1 / (nu + 1);
    double g_alpha = 1 / sqrt(-log(alpha * (2 - alpha)));
    double h_nu_alpha = pow(2 * alpha * sqrt(nu), 1 / nu);
    double t_inverse = -0.0953 - 0.631 * f_nu + 0.81 * g_alpha + 0.076 * h_nu_alpha;
    return 1 / t_inverse;
}

double norm_quantile(double t){
    static double log2 = log(2);
    static double log22 = log(22);
    static double log41 = log(41);

    double alpha = 1 - 1 / t;
    return 10 * log(1 - log(-log(alpha) /log2) / log22) / log41;
}

#endif
