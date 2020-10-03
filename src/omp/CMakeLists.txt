find_package(OpenMP REQUIRED)

add_executable(hello-world hello-world.c)
target_link_libraries(hello-world PUBLIC OpenMP::OpenMP_C)

add_executable(trapezoidal-rule trapezoidal-rule.c)
target_link_libraries(trapezoidal-rule PUBLIC OpenMP::OpenMP_C)
