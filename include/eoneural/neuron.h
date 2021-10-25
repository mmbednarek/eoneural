#pragma once

typedef double* neuron_t;

void neuron_random(neuron_t n, int num_weight);
double neuron_perform(neuron_t n, int num_weight, double *args);
double calc_delta(double eta, double error, double y, double x);
