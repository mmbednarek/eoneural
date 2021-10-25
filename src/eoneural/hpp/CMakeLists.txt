add_library(eoneural_hpp Network.cpp BasicContext.cpp)
target_link_libraries(eoneural_hpp eoneural spdlog fmt)