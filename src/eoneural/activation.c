#include <eoneural/activation.h>
#include <math.h>

double identity(double x, double *params) {
   return x;
}

double identity_deri(double y, double x) {
   return 1.0;
}

double sigmoid(double x, double *params) {
   return 1.0 / (1.0 + exp(-0.8 * x));
}

double sigmoid_deri(double y, double x) {
   return y * (1.0 - y);
}

double _tanh(double x, double *params) {
   return tanh(x);
}

double _tanh_deri(double y, double x) {
   return 1.0 - y * y;
}

double _atan(double x, double *params) {
   return atan(x);
}

double _atan_deri(double y, double x) {
   return 1.0 / (x * x + 1);
}

double gauss(double x, double *params) {
   return exp(-x * x);
}

double gauss_deri(double y, double x) {
   return -2.0 * x * y;
}

struct activation activation_funcs[] = {
        {0, "identity", identity, identity_deri},
        {0, "sigmoid", sigmoid, sigmoid_deri},
        {0, "tanh", _tanh, _tanh_deri},
        {0, "atan", _atan, _atan_deri},
        {0, "gaussian", gauss, gauss_deri},
};
