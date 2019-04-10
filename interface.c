#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "interface.h"
#include "network.h"
#include "activation.h"

static double *load_number_data(const char *filename, int *num) {

    FILE *f;
    size_t size;
    double *data;

    f = fopen(filename, "rb");

    if(!f) {
        fprintf(stderr, "Could not open file.");
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if(size % sizeof(double) != 0) {
        fprintf(stderr, "Invalid size.");
        exit(2);
    }

    data = (double*) malloc(size);
    fread(data, size, 1, f);
    fclose(f);

    *num = size / sizeof(double);

    return data;
}

int action_create(int argc, char **argv) {

    char *output_file = "out.net", *weights_file = NULL;
    double *data;
    int i, k, num, num_in, *num_neurons, num_layers = -1;
    unsigned char activation = 1; // sigmoid default
    network_t net;

    srand(time(0));

    /* Read the params */
    for(i = 2; i < argc; i++) {

        if(!strcmp(argv[i], "-o")) {
            if(++i < argc) {
                output_file = argv[i];
                if(strlen(output_file) == 0) {
                    fprintf(stderr, "File name can't be empty.");
                    return 4;
                }
            }
        } else if(!strcmp(argv[i], "-w")) {
            if(++i < argc) {
                weights_file = argv[i];
                if(strlen(weights_file) == 0) {
                    fprintf(stderr, "File name can't be empty.");
                    return 4;
                }
            }
        } else if(!strcmp(argv[i], "-a")) {
            if(++i < argc) {
                for(k = 0; k < NUM_ACTIVATION_FUNCS; k++) {
                    if(!strcmp(argv[i], activation_funcs[k].name))  {
                        activation = k;
                        break;
                    }
                }
            }
        } else {
            num_layers++;
        }
    }

    if(num_layers < 1) {
        fprintf(stderr, "Needs at least 1 layer.");
        return 1;
    }

    num_neurons = (int*) malloc(sizeof(int) * num_layers);
    num_layers = -1;

    /* Read the arguments*/
    for(i = 2; i < argc; i++) {

        if(argv[i][0] == '-') {
            i++;
        } else {
            if(num_layers == -1) {
                num_in = strtol(argv[i], NULL, 10);
                if(num_in < 1) {
                    fprintf(stderr, "Number of inputs can't be lower than 1.");
                    return 2;
                }
            } else {
                num_neurons[num_layers] = strtol(argv[i], NULL, 10); 
                if(num_neurons[num_layers] < 1) {
                    fprintf(stderr, "Number of neurons in a layer can't be lower than 1.");
                    return 3;
                }
            }
            num_layers++;
        }
    }

    if(!weights_file) {
        net = network_create(num_in, num_layers, num_neurons, activation);
    } else {
        data = load_number_data(weights_file, &num);
        net = network_create_raw(num_in, num_layers, num_neurons, activation, data);
    }
    network_save(net, output_file);

    free(num_neurons);
    free(net);

    return 0;
}

int action_show(int argc, char **argv) {

    network_t net;

    if(argc < 3) {
        fprintf(stderr, "Please provide an input file.");
        return 1;
    }

    net = network_load(argv[2]);

    if(!net) {
        fprintf(stderr,"Invalid network file.");
        return 2;
    }

    printf("Input values: %d\n", net->num_in);
    printf("Output values: %d\n", net->num_out);
    printf("Activation function: %s\n", activation_funcs[net->activation].name);
    printf("Number of layers: %d\n", net->num_layers);
    printf("Total neurons: %d\n", net->total_neurons);
    printf("Total weights: %d\n\n", net->total_weights);

    network_print(net);
    free(net);
    return 0;
}

static void print_array(double *data, int num) {
    int i;
    printf("%lf", data[0]);
    for(i = 1; i < num; i++) {
        printf(" %lf", data[i]);
    }
}

int action_pass(int argc, char **argv) {

    network_t net;
    int i, k, num;
    int cycles;
    char *training = NULL, *testing = NULL;
    double *data;
    double *input, *cursor, *result;

    if(argc < 3) {
        fprintf(stderr, "Please provide an input file.");
        return 1;
    }

    k = 0;
    for(i = 2; i < argc; i++) {
         
        if(!strcmp(argv[i], "-t")) {
            if(++i < argc) {
                training = argv[i];
            }
        } else if(!strcmp(argv[i], "-d")) {
            if(++i < argc) {
                testing = argv[i];
            }
        } else {
            if(k == 0) {
                net = network_load(argv[i]);
                if(!net) {
                    fprintf(stderr, "Invalid network file.");
                    return 2;
                }
                input = (double*) malloc(sizeof(double)*net->num_in);
            } else if(k <= net->num_in) {
                input[k-1] = atof(argv[i]);
            }
            k++;
        }
    }

    if(!net) {
        fprintf(stderr, "Please provide a network file.");
        return 2;
    }

    if(training != NULL) {
        data = load_number_data(training, &num);
        cycles = num / (net->num_in + net->num_out);

        cursor = data;
        for(i = 0; i < cycles; i++) {
            result = network_perform(net, cursor);
            print_array(cursor, net->num_in);
            printf(" → ");
            print_array(result, net->num_out);
            putchar('\n');
            cursor += (net->num_in+net->num_out);
        }

        free(data);
    }

    if(testing != NULL) {
        data = load_number_data(testing, &num);
        cycles = num / (net->num_in);

        cursor = data;
        for(i = 0; i < cycles; i++) {
            result = network_perform(net, cursor);
            print_array(cursor, net->num_in);
            printf(" → ");
            print_array(result, net->num_out);
            putchar('\n');
            cursor += (net->num_in);
        }

        free(data);
    }

    if(k != net->num_in+1) {
        if(!training && !testing) {
            fprintf(stderr, "Wrong number of arguments.");
        }
        return 3;
    }

    result = network_perform(net, input);
    
    printf("%lf", result[0]);
    for(i = 1; i < net->num_out; i++) {
        printf(" %lf", result[i]);
    }
    putchar('\n');

    free(input);
    free(net);
    return 0;
}

int action_pack(int argc, char **argv) {

    FILE *fi, *fo;
    double value;
    int c;
    char number[256]; /* I know I know */
    char *cursor;

    if(argc < 4) {
        fprintf(stderr, "Needs an input and an output file.");
        return 1;
    }

    if(!strcmp(argv[2], argv[3])) {
        fprintf(stderr, "Can't be the same file.");
        return 2;
    }

    if(!strcmp(argv[2], "-")) {
        fi = stdin;
    } else {
        fi = fopen(argv[2], "rb");
    }

    if(!strcmp(argv[3], "-")) {
        fo = stdout;
    } else {
        fo = fopen(argv[3], "wb");
    }


    if(!fi || !fo) {
        fprintf(stderr, "Invalid arguments.");
        return 3;
    }

    cursor = number;

    for(;;) {
        c = fgetc(fi);
        if(feof(fi)) {
            if(cursor != number) {
                *cursor = 0;
                value = atof(number);
                fwrite(&value, sizeof(double), 1, fo); 
            }
            break;
        }
        if((c >= '0' && c <= '9') || c == '-' || c == '.')  {
            *(cursor++) = (char)c;
        } else {
            if(cursor != number) {
                *cursor = 0;
                cursor = number;
                value = atof(number);
                fwrite(&value, sizeof(double), 1, fo); 
            }
        }
    } 

    fclose(fo);
    fclose(fi);

    return 0;
}

int action_train(int argc, char **argv) {

    int i, n, k;
    long long epochs = 0;
    clock_t t_start, t_end;
    char *input_network = NULL, *input_data = NULL, *output_network = NULL;
    double *data, *outputs_partial, *outputs, *errors, *input, *target;
    double mse = 1.0, min_msi = 0.0001;
    double learning = 0.7, momentum = 0.3;
    network_t net;

    k = 0;

    for(i = 2; i < argc; i++) {

        if(!strcmp(argv[i], "-e")) {
            if(++i < argc) {
                epochs = strtol(argv[i], NULL, 10);
            }
        } else if(!strcmp(argv[i], "-l")) {
            if(++i < argc) {
                learning = atof(argv[i]);
            }
        } else if(!strcmp(argv[i], "-m")) {
            if(++i < argc) {
                momentum = atof(argv[i]);
            }
        } else if(!strcmp(argv[i], "-o")) {
            if(++i < argc) {
                output_network = argv[i];
            }
        } else if(!strcmp(argv[i], "-r")) {
            if(++i < argc) {
                min_msi = atof(argv[i]);
                min_msi *= min_msi;
            }
        } else {
            if(k == 0) {
                input_network = argv[i];
                output_network = argv[i];
            } else if(k == 1) {
                input_data = argv[i];
            } else {
                output_network = argv[i];
            }
            k++;
        }
    }

    if(!input_network || !input_data || !output_network) {
        fprintf(stderr, "Not enough files specified.");
        return 1;
    }

    net = network_load(input_network);

    if(!net) {
        fprintf(stderr, "Invalid network file.");
        return 2;
    }

    data = load_number_data(input_data, &n);

    if(!data || n == 0) {
        fprintf(stderr, "Invalid data file.");
        return 2;
    }

    n /= net->num_in + net->num_out;

    outputs_partial = network_alloc_area(net);
    outputs         = network_alloc_area(net);
    errors          = network_alloc_area(net);

    t_start = clock();

    printf("\e[?25l");

    i = 0;
    while(mse > min_msi) {
        input = data;
        mse = 0;
        for(k = 0; k < n; k++) {
            target = input + net->num_in;
            mse += network_train(net, outputs_partial, outputs, errors, input, target, learning, momentum);
            input = target + net->num_out;
        }
        mse /= (double)n;
        printf("EPOCH: %d, RMSE: %lf\r", i, sqrt(mse));

        if(epochs != 0 && i >= epochs)
            break;
        i++;
    }
    printf("\e[?25h");
    putchar('\n');

    t_end = clock();
    printf("Elapsed time: %lfs\n", (double)(t_end - t_start)/CLOCKS_PER_SEC);
    

    free(data);
    free(errors);
    free(outputs);
    free(outputs_partial);

    network_save(net, output_network);

    free(net);

    return 0;
}
