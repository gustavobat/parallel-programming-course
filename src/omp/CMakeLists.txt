find_package(OpenMP REQUIRED)

add_executable(hello-world hello-world.c)
target_link_libraries(hello-world PUBLIC OpenMP::OpenMP_C)
