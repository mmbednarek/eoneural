#ifndef H_EONEURAL_BATCH
#define H_EONEURAL_BATCH

#pragma once

struct network_header;
typedef struct network_header* network_t;

struct batch {
   network_t net;
   double *errors;
   double *output;
   double *partial;
};

typedef struct batch batch_t;

batch_t batch_create(network_t net);
void batch_reset(batch_t batch);
void batch_pass_data(batch_t batch, const double *input, const double *target);
void batch_destroy(batch_t batch);

#endif // H_EONEURAL_BATCH
