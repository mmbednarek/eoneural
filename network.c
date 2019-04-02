#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "network.h"
#include "activation.h"

static void network_calculate_pointers(network_t net) {

    net->num_neurons = (int*) ((char*)net + sizeof(struct network_header));

    net->neurons = (neuron_t) ((char*)net->num_neurons +
                               sizeof(int) * net->num_layers);

    net->deltas = (double*) ((char*)net->neurons + sizeof(double) * net->total_weights);

    net->primary   = (double*) ((char*)net->deltas + sizeof(double) * net->total_weights);
    net->secondary = (double*) ((char*)net->primary + sizeof(double) * net->max_neurons);
}

network_t network_create_raw(int num_in, int num_layers, int *num_neurons, unsigned char activation, double *weights) {

    network_t net;
    neuron_t  neuron;

    int layer,
        num_weight = num_in,
        total_weights   = 0,
        total_neurons   = 0,
        total_weights_d,
        total_neurons_d,
        max_neurons     = 0;

    for(layer = 0; layer < num_layers; layer++) {

        total_weights += num_neurons[layer] * (num_weight+1);
        num_weight = num_neurons[layer];
        total_neurons += num_neurons[layer];

        if(num_neurons[layer] > max_neurons) {
            max_neurons = num_neurons[layer];
        }

        if(layer == num_layers-2) {
            total_weights_d = total_weights;
            total_neurons_d = total_neurons;
        }
    }

    net = malloc(network_size(num_layers, total_weights, max_neurons));

    net->activation      = activation;
    net->num_in          = num_in;
    net->num_layers      = num_layers;
    net->num_out         = num_neurons[num_layers-1];
    net->max_neurons     = max_neurons;
    net->total_neurons   = total_neurons;
    net->total_weights   = total_weights;
    net->total_neurons_d = total_neurons_d;
    net->total_weights_d = total_weights_d;

    network_calculate_pointers(net);

    memcpy(net->num_neurons, num_neurons, num_layers * sizeof(int));

    neuron = net->neurons;
    num_weight = num_in;

    memcpy(net->neurons, weights, sizeof(double) * net->total_weights);
    memset(net->deltas, 0x00, sizeof(double) * net->total_weights);

    return net;
}

network_t network_create(int num_in, int num_layers, int *num_neurons, unsigned char activation) {

    network_t net;
    neuron_t  neuron;

    int layer, n, 
        num_weight = num_in,
        total_weights   = 0,
        total_neurons   = 0,
        total_weights_d,
        total_neurons_d,
        max_neurons     = 0;

    for(layer = 0; layer < num_layers; layer++) {

        total_weights += num_neurons[layer] * (num_weight+1);
        num_weight = num_neurons[layer];
        total_neurons += num_neurons[layer];

        if(num_neurons[layer] > max_neurons) {
            max_neurons = num_neurons[layer];
        }

        if(layer == num_layers-2) {
            total_weights_d = total_weights;
            total_neurons_d = total_neurons;
        }
    }

    net = malloc(network_size(num_layers, total_weights, max_neurons));

    net->activation      = activation;
    net->num_in          = num_in;
    net->num_layers      = num_layers;
    net->num_out         = num_neurons[num_layers-1];
    net->max_neurons     = max_neurons;
    net->total_neurons   = total_neurons;
    net->total_weights   = total_weights;
    net->total_neurons_d = total_neurons_d;
    net->total_weights_d = total_weights_d;

    network_calculate_pointers(net);

    memcpy(net->num_neurons, num_neurons, num_layers * sizeof(int));

    neuron = net->neurons;
    num_weight = num_in;

    foreach_layer(net, layer, num_weight) {
        foreach_neuron(net, n, neuron, layer, num_weight) {
            neuron_random(neuron, num_weight);
        }
    }

    memset(net->deltas, 0x00, sizeof(double) * net->total_weights);

    return net;
}

#define APP_VERSION 1
static const char NETWORK_SIG[] = "NNT";

void network_save(network_t net, const char *filename) {

    void *num_neurons, *neurons, *deltas, *primary, *secondary;
    FILE *f;
    struct network_file_header header;

    memcpy(header.signature, NETWORK_SIG, 3);
    header.version = APP_VERSION;
    header.network_size = network_size(net->num_layers, net->total_weights, net->max_neurons);

    /* we don't want to save pointers */
    num_neurons = net->num_neurons;
    neurons = net->neurons;
    deltas = net->deltas;
    primary = net->primary;
    secondary = net->secondary;
    net->num_neurons = NULL;
    net->neurons     = NULL;
    net->deltas      = NULL;
    net->primary     = NULL;
    net->secondary   = NULL;

    f = fopen(filename, "wb");

    if(!f) {
        return;
    }

    fwrite(&header, sizeof(struct network_file_header), 1, f);
    fwrite(net, header.network_size, 1, f);
    fclose(f);

    /* bring back the pointers */
    net->num_neurons = num_neurons;
    net->neurons     = neurons;
    net->deltas      = deltas;
    net->primary     = primary;
    net->secondary   = secondary;
}

network_t network_load(const char *filename) {

    FILE *f;
    struct network_file_header header;
    network_t net;

    f = fopen(filename, "rb");

    if(!f) {
        return NULL;
    }

    fread(&header, sizeof(struct network_file_header), 1, f);

    if(header.signature[0] != NETWORK_SIG[0] ||
       header.signature[1] != NETWORK_SIG[1] ||
       header.signature[2] != NETWORK_SIG[2]) {

        return NULL;
    }

    if(header.version > APP_VERSION) {
        return NULL;
    }

    if(header.network_size < sizeof(struct network_header))  {
        return NULL;
    }

    net = malloc(header.network_size);
    fread(net, header.network_size, 1, f);
    fclose(f);

    network_calculate_pointers(net);

    return net;
}

void network_print(network_t net) {

    int num_weight, layer, n, i;
    neuron_t neuron = net->neurons;
    double *deltas = net->deltas;

    foreach_layer(net, layer, num_weight) {

        printf("LAYER %d\n", layer);
        foreach_neuron(net, n, neuron, layer, num_weight) {

            printf("\tNeuron %d: \n", n);

            for(i = 0; i <= num_weight; i++) {
                printf("\t\tW%d: %lf, ΔW%d: %lf\n", i, neuron[i], i, deltas[i]);
            }

            deltas += num_weight + 1;
        }
    }
}

static void network_swap(network_t net) {

    double *tmp;

    tmp = net->primary;
    net->primary = net->secondary;
    net->secondary = tmp;
}


double *network_perform(network_t net, double *input) {

    int num_weight, layer, n;
    neuron_t neuron;
    double(*act_func)(double,double*);

    memcpy(net->primary, input, sizeof(double) * net->num_in);

    neuron = net->neurons;
    act_func = activation_funcs[net->activation].func;

    foreach_layer(net, layer, num_weight) {

        foreach_neuron(net, n, neuron, layer, num_weight) {
            net->secondary[n] = act_func(neuron_perform(neuron, num_weight, net->primary), NULL);
        }

        network_swap(net);
    }

    return net->primary; /*Return value is not copied!*/
}

static double *network_perform_ex(network_t net, double *input, double *output_partial, double *output) {

    int num_weight, layer, n, neuron_index;
    neuron_t neuron;
    double(*act_func)(double,double*);

    memcpy(net->primary, input, sizeof(double) * net->num_in);

    neuron_index = 0;
    neuron = net->neurons;
    act_func = activation_funcs[net->activation].func;

    foreach_layer(net, layer, num_weight) {

        foreach_neuron(net, n, neuron, layer, num_weight) {
            output_partial[neuron_index] = neuron_perform(neuron, num_weight, net->primary);
            net->secondary[n] = act_func(output_partial[neuron_index], NULL);
            output[neuron_index] = net->secondary[n];
            neuron_index++;
        }

        network_swap(net);
    }

    return net->primary;
}

double *network_alloc_area(network_t net) {
    return (double*) malloc(sizeof(double) * net->total_neurons);
}

static void network_backprop(network_t net, double *partial, double *outputs, double *errors) {

    int layer, n, i, num_weight, neuron_index, neuron_calc_index;
    double(*act_func_deri)(double,double);
    neuron_t neuron = net->neurons + net->total_weights;
    act_func_deri = activation_funcs[net->activation].deri;

    if(net->num_layers < 2) {
        return; /* No need for backprop only one layer */
    }

    neuron_calc_index = net->total_neurons_d;
    neuron_index      = net->total_neurons;

    foreach_layer_rev(net, layer, num_weight) {

        neuron_calc_index -= num_weight;

        // Zero the errors
        for(i = 0; i < num_weight; i++) {
            errors[neuron_calc_index+i] = 0.0;
        }

        // For each neuron reversed
        for(n = net->num_neurons[layer]-1; n >= 0; n--) {
            neuron -= num_weight + 1;
            neuron_index--;

            for(i = 0; i < num_weight; i++) {
                errors[neuron_calc_index+i] += errors[neuron_index] * neuron[i+1] * act_func_deri(outputs[neuron_calc_index+i], partial[neuron_calc_index+i]);
            }

        }
    }
}

double network_train(network_t net, double *partial, double *outputs, double *errors, double *input, double *target, double learning, double momentum) {

    int i, layer, n, num_weight, neuron_index;
    double *ol_outputs, *ol_errors, *ol_partial, *neuron_input, *tmp;
    double mse = 0.0;
    neuron_t neuron;
    double *deltas;
    double(*act_func_deri)(double,double);
    act_func_deri = activation_funcs[net->activation].deri;

    ol_outputs = network_perform_ex(net, input, partial, outputs);
    ol_partial = partial + (net->total_neurons - net->num_out);
    ol_errors  = errors + (net->total_neurons - net->num_out);

    for(i = 0; i < net->num_out; i++) {
        ol_errors[i] = ol_outputs[i] - target[i];
        mse += ol_errors[i] * ol_errors[i];
        ol_errors[i] *= act_func_deri(ol_outputs[i], ol_partial[i]);
    }

    mse /= net->num_out;

    network_backprop(net, partial, outputs, errors);


    /*
    printf("TRAINING ");
    printf("INPUT (%lf", input[0]);
    for(i = 1; i < net->num_in; i++) {
        printf(", %lf", input[i]);
    }
    printf(") TARGET (%lf", target[0]);
    for(i = 1; i < net->num_out; i++) {
        printf(", %lf", input[i]);
    }
    printf(")\n");*/

    neuron_index = 0;
    neuron = net->neurons;
    deltas = net->deltas;
    neuron_input = input;
    foreach_layer(net, layer, num_weight) {

        tmp = outputs + neuron_index;

        //printf("\tLAYER %d\n", layer);

        foreach_neuron(net, n, neuron, layer, num_weight) {

            //printf("\t\tNeuron %d (Output %lf, Error: %lf): \n", n, outputs[neuron_index], errors[neuron_index]);
            //printf("\t\t\tW0:  %lf, ΔE: %lf\n", neuron[0], eta * errors[neuron_index] * act_func_deri(outputs[neuron_index], outputs_partial[neuron_index]));

            deltas[0] *= momentum; 
            deltas[0] -= learning * errors[neuron_index];
            neuron[0] += deltas[0];

            for(i = 1; i <= num_weight; i++) {
                //printf("\t\t\tW%d:  %lf, ΔE: %lf\n", i, neuron[i], eta * errors[neuron_index] * act_func_deri(outputs[neuron_index], outputs_partial[neuron_index]) * neuron_input[i-1]);
                deltas[i] *= momentum; 
                deltas[i] -= learning * errors[neuron_index] * neuron_input[i-1];
                neuron[i] += deltas[i];
            }

            neuron_index++;
            deltas += num_weight + 1;
        }

        neuron_input = tmp;
    }

    return mse;

}
