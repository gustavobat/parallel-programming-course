enable_language(CUDA)

add_executable(hello-world-cuda hello-world.cu common.h)
add_executable(int-add int-add.cu common.h)
add_executable(device-query device-query.cu common.h)
add_executable(long-vector-add long-vector-add.cu common.h)
