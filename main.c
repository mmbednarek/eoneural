#include <stdio.h>
#include <string.h>
#include "interface.h"

int main(int argc, char **argv) {
    if(argc < 2) {
        puts("Please provide an action. (random/show/pack/train/pass)");
        return 1;
    }

    if(!strcmp(argv[1], "create")) {
        return action_create(argc, argv);
    } else if(!strcmp(argv[1], "show")) {
        return action_show(argc, argv);
    } else if(!strcmp(argv[1], "pack")) {
        return action_pack(argc, argv);
    } else if(!strcmp(argv[1], "train")) {
        return action_train(argc, argv);
    } else if(!strcmp(argv[1], "pass")) {
        return action_pass(argc, argv);
    } 

    printf("Unknown option %s\n", argv[1]);
    return 5;
}
