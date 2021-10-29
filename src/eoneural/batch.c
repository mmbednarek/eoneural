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

void batch_reset(batch_t batch) {
   memset(batch.errors, 0, batch.net->total_weights);
}

void batch_pass_data(batch_t batch, const double *input, const double *target) {
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


   foreach_layer(net, layer, num_weight) {
      foreach_neuron(net, n, neuron, layer, num_weight) {
      }
   }
}

void batch_destroy(batch_t batch) {
   free(batch.partial);
   free(batch.output);
   free(batch.errors);
}