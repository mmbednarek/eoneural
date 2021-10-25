add_library(eoneural activation.c network.c neuron.c)
target_include_directories(eoneural PUBLIC ../../include)
target_link_libraries(eoneural PUBLIC m)
