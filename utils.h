#pragma once
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

double random_weight() {
    return (double) (rand() % 200 - 100) / 100.0;
}
