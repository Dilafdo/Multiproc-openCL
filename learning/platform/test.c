#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>

#include <opencl_include.h>

int main() {

    cl_platform_id *platforms;
    cl_uint num_platforms, j;
    cl_int err;
    // cl_ulong mem_size;

    // mem_size = 23874359374L;
    // printf("Memory size: %lu\n", mem_size);

    err = clGetPlatformIDs(1, NULL, &num_platforms);
    if (err < 0) {
        perror("Error: clGetPlatformIDs");
        exit(1);
    }

    printf("Number of platforms: %u\n", num_platforms);
    printf("\n");

    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);

    clGetPlatformIDs(num_platforms, platforms, NULL);

    for (j = 0; j < num_platforms; j++) {

        char p_name[40];
        char p_vendor[40];
        err = clGetPlatformInfo(platforms[j], CL_PLATFORM_NAME, 40, p_name, NULL);
        if (err < 0) {
            perror("Error: clGetPlatformInfo CL_PLATFORM_NAME");
            exit(1);
        }
        err = clGetPlatformInfo(platforms[j], CL_PLATFORM_VENDOR, 40, p_vendor, NULL);
        if (err < 0) {
            perror("Error: clGetPlatformInfo CL_PLATFORM_VENDOR");
            exit(1);
        }

        cl_uint num_devices, i;

        err = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (err < 0) {
            perror("Error: clGetDeviceIDs");
            exit(1);
        }

        cl_device_id* devices = calloc(sizeof(cl_device_id), num_devices);
        clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        char buf[128];
        cl_uint work_item_dim, compute_units, char_vector_width, global_mem_size, global_mem_cache, buffer_size, local_mem_size;
        cl_bool image_support;
        size_t time_res, local_size;
        cl_device_fp_config flag;

        if (num_devices == 0) {
            printf("no device found!");
        }

        printf("Number of devices: %u\n", num_devices);
        printf("\n");

        for (i = 0; i < num_devices; i++) {
            clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, buf, NULL);
            fprintf(stdout, "Platform:  %s\n", p_name);
            fprintf(stdout, "Vendor:    %s\n", p_vendor);
            fprintf(stdout, "Device:    %s\n",buf);
            clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 128, buf, NULL);
            fprintf(stdout, "Device OpenCL version : %s\n", buf);

            clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &work_item_dim, NULL);
            fprintf(stdout, "Maximum work dimensions device support : \t%u\n", work_item_dim);

            clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
            fprintf(stdout, "The number of parallel compute units : \t\t%u\n", compute_units);

            clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_size, NULL);
            fprintf(stdout, "Maximum work group size : \t\t\t%zu \n", local_size);

            clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(char_vector_width), &char_vector_width, NULL);
            fprintf(stdout, "Preferred native vector width size : \t\t%u\n", char_vector_width);

            clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_uint), &global_mem_size, NULL);
            fprintf(stdout, "Max size of memory object allocation in bytes : %u\n", global_mem_size);

            clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_uint), &global_mem_cache, NULL);
            fprintf(stdout, "Size of global memory cache in bytes : \t\t%u\n", global_mem_cache);

            clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_uint), &buffer_size, NULL);
            fprintf(stdout, "Max constant buffer allocation in bytes : \t%u\n", buffer_size);

            clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_uint), &local_mem_size, NULL);
            fprintf(stdout, "Size of local memory region in bytes :  \t%u\n", local_mem_size);

            clGetDeviceInfo(devices[i], CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(time_res), &time_res, NULL);
            fprintf(stdout, "Resolution of device timer : \t\t\t%zu ns\n", time_res);

            clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
            if (image_support == CL_TRUE) {
                printf("Device Support Image operations!\n");
            } else {
                printf("Device doesn't support Image operations!\n");
            }

            // Query the maximum dimensions for 2D images
            size_t max_width, max_height;
            err = clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &max_width, NULL);
            if (err != CL_SUCCESS) {
                printf("Error getting max width\n");
                return 1;
            }

            err = clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &max_height, NULL);
            if (err != CL_SUCCESS) {
                printf("Error getting max height\n");
                return 1;
            }

            printf("Maximum supported 2D Image Width: %zu\n", max_width);
            printf("Maximum supported 2D Image Height: %zu\n", max_height);

            printf("\n");

            printf("=======================================\n");
            printf("\n");

            err = clGetDeviceInfo(devices[i], CL_DEVICE_SINGLE_FP_CONFIG, sizeof(flag), &flag, NULL);
            if(err < 0) {
                perror("Couldn't read device FP information");
                exit(1);
            }
            printf("Float Processing Features:\n");
            if(flag & CL_FP_INF_NAN)
            printf("INF and NaN values supported.\n");
            if(flag & CL_FP_DENORM)
            printf("Denormalized numbers supported.\n");
            if(flag & CL_FP_ROUND_TO_NEAREST)
            printf("Round To Nearest Even mode supported.\n");
            if(flag & CL_FP_ROUND_TO_INF)
            printf("Round To Infinity mode supported.\n");
            if(flag & CL_FP_ROUND_TO_ZERO)
            printf("Round To Zero mode supported.\n");
            if(flag & CL_FP_FMA)
            printf("Floating-point multiply-and-add operation supported.\n");
        }

        free(devices);
    }
    free(platforms);

    return 0;
}