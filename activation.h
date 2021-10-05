#pragma once

struct activation {
    unsigned num_params;
    const char *name;
    double(*func)(double,double*);
    double(*deri)(double,double);

};

#define NUM_ACTIVATION_FUNCS 5
extern struct activation activation_funcs[NUM_ACTIVATION_FUNCS];
