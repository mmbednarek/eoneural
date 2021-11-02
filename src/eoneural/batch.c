#include <eoneural/activation.h>
#include <eoneural/batch.h>
#include <eoneural/network.h>
#include <stdlib.h>
#include <string.h>

batch_t batch_create(network_t net) {
   batch_t result = {
           .net = net,
           .errors = network_alloc_area(net),
           .output = network_alloc_area(net),
           .partial = network_alloc_area(net),
   };
   memset(result.errors, 0, net->total_weights);
   return result;
}

void batch_begin(batch_t batch, double momentum) {
   for (size_t i = 0; i < batch.net->total_weights; ++i) {
      batch.net->deltas[i] *= momentum;
   }
}

void batch_put(batch_t batch, const double *input, const double *target, double learning) {
   double *ol_outputs = network_perform_ex(batch.net, input, batch.partial, batch.output);

   double (*act_func_deri)(double, double) = activation_funcs[batch.net->activation].deri;

   size_t last_layer_index = batch.net->total_neurons - batch.net->num_out;
   double *ol_partial = batch.partial + last_layer_index;
   double *ol_errors = batch.errors + last_layer_index;

   for (size_t i = 0; i < batch.net->num_out; i++) {
      ol_errors[i] = ol_outputs[i] - target[i];
      ol_errors[i] *= act_func_deri(ol_outputs[i], ol_partial[i]);
   }

   network_backprop(batch.net, batch.partial, batch.output, batch.errors);

   size_t neuron_index = 0;
   size_t layer = 0;
   size_t num_weight = 0;
   neuron_t neuron = batch.net->neurons;
   size_t n;
   double *deltas = batch.net->deltas;
   const double *neuron_input = input;

   foreach_layer(batch.net, layer, num_weight) {
      double *tmp = batch.output + neuron_index;

      foreach_neuron(batch.net, n, neuron, layer, num_weight) {
         deltas[0] -= learning * batch.errors[neuron_index];

         for (size_t i = 1; i <= num_weight; i++) {
            deltas[i] -= learning * batch.errors[neuron_index] * neuron_input[i - 1];
         }

         neuron_index++;
         deltas += num_weight + 1;
      }

      neuron_input = tmp;
   }
}


void batch_end(batch_t batch) {
   neuron_t neuron = batch.net->neurons;
   double *deltas = batch.net->deltas;
   size_t n;
   size_t layer = 0;
   size_t num_weight = 0;

   foreach_layer(batch.net, layer, num_weight) {
      foreach_neuron(batch.net, n, neuron, layer, num_weight) {
         neuron[0] += deltas[0];

         for (size_t i = 1; i <= num_weight; i++) {
            neuron[i] += deltas[i];
         }

         deltas += num_weight + 1;
      }
   }
}

void batch_destroy(batch_t batch) {
   if (batch.partial != NULL)
      free(batch.partial);
   if (batch.output != NULL)
      free(batch.output);
   if (batch.errors != NULL)
      free(batch.errors);
}