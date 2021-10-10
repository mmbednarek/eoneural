cmake_minimum_required(VERSION 3.15)
project(eoneural)

add_executable(eoneural activation.c interface.c main.c network.c neuron.c)
target_link_libraries(eoneural m)
