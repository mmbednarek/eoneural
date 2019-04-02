#include <stdlib.h>
#include "utils.h"
#include "neuron.h"

void neuron_random(neuron_t n, int num_weight) {
    int i;

    for(i = 0; i <= num_weight; i++) {
        n[i] = random_weight();
    }
}

double neuron_perform(neuron_t n, int num_weight, double *args) {
    int i;
    double result = n[0];

    for(i = 1; i <= num_weight; i++) {
        result += args[i-1] * n[i];
    }

    return result;
}
