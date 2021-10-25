add_executable(eoneural_test TrainTest.cpp)
target_link_libraries(eoneural_test eoneural_hpp ${GTEST_LIBRARIES} gtest_main)