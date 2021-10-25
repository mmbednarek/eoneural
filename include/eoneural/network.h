#pragma once
#include "neuron.h"
#include <stddef.h>

struct network_header {
   int num_in;
   int num_layers;

   /* Network params */
   unsigned char activation;

   /* Redundant variables */
   int num_out;
   int max_neurons;
   int total_neurons;
   int total_weights;
   int total_neurons_d; /* Disregard output layer */
   int total_weights_d; /* Disregard output layer */

   /* Redundant pointers */
   int *num_neurons;
   neuron_t neurons;
   double *deltas;
   double *primary; /* Useful space for propagation */
   double *secondary;
};

struct network_file_header {
   char signature[3];
   short version;
   size_t network_size;
};

typedef struct network_header *network_t;

#define network_size(num_layers, total_weights, max_neurons) \
   sizeof(struct network_header) +                           \
           sizeof(int) * num_layers +                        \
           sizeof(double) * total_weights * 2 +              \
           sizeof(double) * max_neurons * 2

#define foreach_layer(net, index, num_weight) for (index = 0, num_weight = net->num_in; \
                                                   index < net->num_layers;             \
                                                   num_weight = net->num_neurons[index++])

#define foreach_neuron(net, index, neuron, layer, num_weight) for (index = 0;                       \
                                                                   index < net->num_neurons[layer]; \
                                                                   index++, neuron += num_weight + 1)

#define foreach_layer_rev(net, index, num_weight) for (index = net->num_layers - 1, num_weight = net->num_neurons[index - 1]; \
                                                       index > 0;                                                             \
                                                       index--, num_weight = net->num_neurons[index - 1])

network_t network_create_raw(int num_in, int num_layers, int *num_neurons, unsigned char activation, double *weights);
network_t network_create(int num_in, size_t num_layers, int *num_neurons, unsigned char activation);
double *network_perform(network_t net, const double *input);
double network_train(network_t net, double *partial, double *outputs, double *errors, const double *input, const double *target, double learning, double momentum);
double *network_alloc_area(network_t net);
void network_save(network_t net, const char *filename);
network_t network_load(const char *filename);
void network_print(network_t net);
void network_destroy(network_t net);
network_t network_copy(network_t net);