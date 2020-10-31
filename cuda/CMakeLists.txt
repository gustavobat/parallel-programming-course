enable_language(CUDA)

add_executable(hello-world-cuda hello-world.cu)
target_compile_options(hello-world-cuda PRIVATE "-arch=sm_35")
set_property(TARGET hello-world-cuda PROPERTY CUDA_ARCHITECTURES OFF)

add_executable(int-add int-add.cu)
target_compile_options(int-add PRIVATE "-arch=sm_35")
set_property(TARGET int-add PROPERTY CUDA_ARCHITECTURES OFF)
