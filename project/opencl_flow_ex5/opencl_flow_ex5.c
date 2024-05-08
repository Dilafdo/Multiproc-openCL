#define KERNEL_RESIZE_IMAGE "resize_image"
#define KERNEL_COLOR_TO_GRAY "color_to_gray"
#define KERNEL_ZNCC_LEFT "left_disparity"
#define KERNEL_ZNCC_RIGHT "right_disparity"
#define KERNEL_CROSS_CHECK "crosscheck"
#define KERNEL_OCCLUSION_FILL "occlusion_fill"

#include <pngloader.h>
#include <config.h>

#include <opencl_flow_ex5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config_im_to_g.h"

cl_device_id create_device(void) {

   cl_platform_id platform;
   cl_device_id device;
   int err;

   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   }

   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);
   }

   return device;
}

cl_program build_program(cl_context ctx, cl_device_id device, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   program = clCreateProgramWithSource(ctx, 1,
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

cl_ulong getExecutionTime(cl_event event) {
    cl_ulong start_time, end_time;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
    return end_time - start_time;
}

void resize_image(cl_context context, cl_kernel kernel, cl_command_queue queue, const Image *im0, Image *output_im0) {

    /* Image data */
    cl_mem input_image, output_image;
    cl_image_format input_format, output_format;
    int err;

    cl_ulong read_time, time_to_downsample;

    cl_event resize_read_event, resize_event;

    size_t width = im0 -> width;
    size_t height = im0 -> height;
    size_t new_width = width / 4;
    size_t new_height = height / 4;

    input_format.image_channel_order = CL_RGBA;
    input_format.image_channel_data_type = CL_UNORM_INT8;

    output_format.image_channel_order = CL_RGBA;
    output_format.image_channel_data_type = CL_UNORM_INT8;

    /* Create input image object */
    input_image = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &input_format, width, height, 0, (void*)im0 -> image, &err);
    if(err < 0) {
        printf("resize_image: Couldn't create the input image object");
        exit(1);
    };

    /* Create output image object */
    output_image = clCreateImage2D(context, CL_MEM_READ_WRITE, &output_format, new_width, new_height, 0, NULL, &err);
    if(err < 0) {
        perror("resize_image: Couldn't create the input image object");
        exit(1);
    };

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    if(err < 0) {
        perror("resize_image, Error: clSetKernelArg, inputImage");
        exit(1);
    }

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
    if(err < 0) {
        perror("resize_image, Error: clSetKernelArg, outputImage");
        exit(1);
    }

    // Execute the OpenCL kernel
    size_t globalWorkSize[2] = { new_width, new_height };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &resize_event);
    if(err < 0) {
        perror("resize_image, Error: clEnqueueNDRangeKernel");
        exit(1);
    }

    // Read the output image back to the host
    err = clEnqueueReadImage(queue, output_image, CL_TRUE, (size_t[3]){0, 0, 0}, (size_t[3]){new_width, new_height, 1},
                             0, 0, (void*)output_im0 -> image, 0, NULL, &resize_read_event);
    if(err < 0) {
        perror("resize_image, Error: clEnqueueReadImage");
        exit(1);
    }

    clFinish(queue);

    output_im0 -> width = new_width;
    output_im0 -> height = new_height;

    time_to_downsample = getExecutionTime(resize_event);
    read_time = getExecutionTime(resize_read_event);

    clReleaseEvent(resize_read_event);
    clReleaseEvent(resize_event);

    printf("Time taken to do the downsampling = %llu ns\n", time_to_downsample);
    printf("Time taken to read the output image (downsampling) = %llu ns\n", read_time);
}

void convert_image_to_gray(cl_context context, cl_kernel kernel, cl_command_queue queue, const Image *im0, Image *output_im0) {

    /* Image data */
    cl_mem input_image, output_image;
    cl_image_format input_format, output_format;
    int err;

    cl_ulong read_time, time_to_grayscale;

    cl_event grayscale_read_event, grayscale_event;

    size_t width = im0 -> width;
    size_t height = im0 -> height;

    input_format.image_channel_order = CL_RGBA;
    input_format.image_channel_data_type = CL_UNORM_INT8;

    output_format.image_channel_order = CL_RGBA;
    output_format.image_channel_data_type = CL_UNORM_INT8;

    /* Create input image object */
    input_image = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &input_format, width, height, 0, (void*)im0 -> image, &err);
    if(err < 0) {
        printf("convert_image_to_gray: Couldn't create the input image object");
        exit(1);
    };

    /* Create output image object */
    output_image = clCreateImage2D(context, CL_MEM_READ_WRITE, &output_format, width, height, 0, NULL, &err);
    if(err < 0) {
        perror("convert_image_to_gray: Couldn't create the input image object");
        exit(1);
    };

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    if(err < 0) {
        perror("convert_image_to_gray, Error: clSetKernelArg, inputImage");
        exit(1);
    }

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
    if(err < 0) {
        perror("convert_image_to_gray, Error: clSetKernelArg, outputImage");
        exit(1);
    }

    // Execute the OpenCL kernel
    size_t globalWorkSize[2] = { width, height };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &grayscale_event);
    if(err < 0) {
        perror("convert_image_to_gray, Error: clEnqueueNDRangeKernel");
        exit(1);
    }

    // Read the output image back to the host
    err = clEnqueueReadImage(queue, output_image, CL_TRUE, (size_t[3]){0, 0, 0}, (size_t[3]){width, height, 1},
                             0, 0, (void*)output_im0 -> image, 0, NULL, &grayscale_read_event);
    if(err < 0) {
        perror("convert_image_to_gray, Error: clEnqueueReadImage");
        exit(1);
    }

    clFinish(queue);

    output_im0 -> width = width;
    output_im0 -> height = height;

    time_to_grayscale = getExecutionTime(grayscale_event);
    read_time = getExecutionTime(grayscale_read_event);

    clReleaseEvent(grayscale_read_event);
    clReleaseEvent(grayscale_event);

    printf("Time taken to do the gray scaling = %llu ns\n", time_to_grayscale);
    printf("Time taken to read the output image (gray scaling) = %llu ns\n", read_time);
}

void apply_zncc(cl_device_id device, cl_context context, cl_kernel kernel, cl_command_queue queue, const Image *im0, const Image *im1, Image *output_im0) {

    size_t wg_size, wg_multiple;
    cl_ulong private_usage, local_usage;

    /* Image data */
    cl_mem input_image, input_image1, output_image;
    cl_image_format input_format, input_format1, output_format;
    int err;

    cl_ulong read_time, time_to_gaussian_blur;
    
    cl_event gaussian_read_event, gaussian_event;

    const size_t width = im0 -> width;
    const size_t height = im0 -> height;

    input_format.image_channel_order = CL_RGBA;
    input_format.image_channel_data_type = CL_UNORM_INT8;

    input_format1.image_channel_order = CL_RGBA;
    input_format1.image_channel_data_type = CL_UNORM_INT8;

    output_format.image_channel_order = CL_RGBA;
    output_format.image_channel_data_type = CL_UNORM_INT8;

    /* Create input image object */
    input_image = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &input_format, width, height, 0, (void*)im0 -> image, &err);
    if(err < 0) {
        printf("zncc: Couldn't create the input image 0 object");
        exit(1);
    };

    input_image1 = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &input_format1, width, height, 0, (void*)im1 -> image, &err);
    if(err < 0) {
        printf("zncc: Couldn't create the input image 1 object");
        exit(1);
    };

    /* Create output image object */
    output_image = clCreateImage2D(context, CL_MEM_READ_WRITE, &output_format, width, height, 0, NULL, &err);
    if(err < 0) {
        perror("zncc: Couldn't create the input image object");
        exit(1);
    };

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    if(err < 0) {
        perror("zncc, Error: clSetKernelArg, inputImage");
        exit(1);
    }

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_image1);
    if(err < 0) {
        perror("zncc, Error: clSetKernelArg, inputImage");
        exit(1);
    }

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_image);
    if(err < 0) {
        perror("zncc, Error: clSetKernelArg, outputImage");
        exit(1);
    }

    // size_t localMemSize = 32 * 32 * sizeof(cl_float4);
    // err = clSetKernelArg(kernel, 3, localMemSize, NULL);
    // if(err < 0) {
    //     perror("zncc, Error: clSetKernelArg, localMemSize 1");
    //     exit(1);
    // }

    // err = clSetKernelArg(kernel, 4, localMemSize, NULL);
    // if(err < 0) {
    //     perror("zncc, Error: clSetKernelArg, localMemSize 2");
    //     exit(1);
    // }

    /* Access kernel/work-group properties */
    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg_size), &wg_size, NULL);
    err |= clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(wg_multiple), &wg_multiple, NULL);
    err |= clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(local_usage), &local_usage, NULL);
    err |= clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(private_usage), &private_usage, NULL);
    if(err < 0) {
        perror("Couldn't obtain kernel work-group size information");
        exit(1);
    };

    printf("The maximum work-group size is %zu and the work-group multiple is %zu.\n\n", wg_size, wg_multiple);
    printf("The kernel uses %zu bytes of local memory. It uses %zu bytes of private memory.\n", 
         local_usage, private_usage);

    // Execute the OpenCL kernel
    size_t globalWorkSize[2] = { width, height };
    // size_t localWorkSize[2] = {32, 32};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &gaussian_event);
    if(err < 0) {
        perror("zncc, Error: clEnqueueNDRangeKernel");
        exit(1);
    }

    // Read the output image back to the host
    err = clEnqueueReadImage(queue, output_image, CL_TRUE, (size_t[3]){0, 0, 0}, (size_t[3]){width, height, 1},
                             0, 0, (void*)output_im0 -> image, 0, NULL, &gaussian_read_event);
    if(err < 0) {
        perror("zncc, Error: clEnqueueReadImage");
        exit(1);
    }

    clFinish(queue);

    output_im0 -> width = width;
    output_im0 -> height = height;

    time_to_gaussian_blur = getExecutionTime(gaussian_event);
    read_time = getExecutionTime(gaussian_read_event);

    clReleaseEvent(gaussian_read_event);
    clReleaseEvent(gaussian_event);

    printf("Time taken to do the zncc = %llu ns\n", time_to_gaussian_blur);
    printf("Time taken to read the output image (zncc) = %llu ns\n", read_time);
}

void apply_crosscheck(cl_context context, cl_kernel kernel, cl_command_queue queue, const Image *im0, const Image *im1, Image *output_im0) {

    /* Image data */
    cl_mem input_image, input_image1, output_image;
    cl_image_format input_format, input_format1, output_format;
    int err;

    cl_ulong read_time, time_to_crosscheck;
    
    cl_event crosscheck_read_event, crosscheck_event;

    const size_t width = im0 -> width;
    const size_t height = im0 -> height;

    input_format.image_channel_order = CL_RGBA;
    input_format.image_channel_data_type = CL_UNORM_INT8;

    input_format1.image_channel_order = CL_RGBA;
    input_format1.image_channel_data_type = CL_UNORM_INT8;

    output_format.image_channel_order = CL_RGBA;
    output_format.image_channel_data_type = CL_UNORM_INT8;

    /* Create input image object */
    input_image = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &input_format, width, height, 0, (void*)im0 -> image, &err);
    if(err < 0) {
        printf("crosscheck: Couldn't create the input image 0 object");
        exit(1);
    };

    input_image1 = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &input_format1, width, height, 0, (void*)im1 -> image, &err);
    if(err < 0) {
        printf("crosscheck: Couldn't create the input image 1 object");
        exit(1);
    };

    /* Create output image object */
    output_image = clCreateImage2D(context, CL_MEM_READ_WRITE, &output_format, width, height, 0, NULL, &err);
    if(err < 0) {
        perror("crosscheck: Couldn't create the input image object");
        exit(1);
    };

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    if(err < 0) {
        perror("crosscheck, Error: clSetKernelArg, inputImage");
        exit(1);
    }

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_image1);
    if(err < 0) {
        perror("crosscheck, Error: clSetKernelArg, inputImage");
        exit(1);
    }

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_image);
    if(err < 0) {
        perror("crosscheck, Error: clSetKernelArg, outputImage");
        exit(1);
    }

    // Execute the OpenCL kernel
    size_t globalWorkSize[2] = { width, height };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &crosscheck_event);
    if(err < 0) {
        perror("crosscheck, Error: clEnqueueNDRangeKernel");
        exit(1);
    }

    // Read the output image back to the host
    err = clEnqueueReadImage(queue, output_image, CL_TRUE, (size_t[3]){0, 0, 0}, (size_t[3]){width, height, 1},
                             0, 0, (void*)output_im0 -> image, 0, NULL, &crosscheck_read_event);
    if(err < 0) {
        perror("crosscheck, Error: clEnqueueReadImage");
        exit(1);
    }

    clFinish(queue);

    output_im0 -> width = width;
    output_im0 -> height = height;

    time_to_crosscheck = getExecutionTime(crosscheck_event);
    read_time = getExecutionTime(crosscheck_read_event);

    clReleaseEvent(crosscheck_read_event);
    clReleaseEvent(crosscheck_event);

    printf("Time taken to do the crosscheck = %llu ns\n", time_to_crosscheck);
    printf("Time taken to read the output image (crosscheck) = %llu ns\n", read_time);
}

void apply_occlusion_fill(cl_context context, cl_kernel kernel, cl_command_queue queue, const Image *im0, Image *output_im0) {

    /* Image data */
    cl_mem input_image, output_image;
    cl_image_format input_format, output_format;
    int err;

    cl_ulong read_time, time_to_occlustion_fill;
    
    cl_event occlustion_fill_read_event, occlustion_fill_event;

    const size_t width = im0 -> width;
    const size_t height = im0 -> height;

    input_format.image_channel_order = CL_RGBA;
    input_format.image_channel_data_type = CL_UNORM_INT8;

    output_format.image_channel_order = CL_RGBA;
    output_format.image_channel_data_type = CL_UNORM_INT8;

    /* Create input image object */
    input_image = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &input_format, width, height, 0, (void*)im0 -> image, &err);
    if(err < 0) {
        printf("occlustion_fill: Couldn't create the input image 0 object");
        exit(1);
    };

    /* Create output image object */
    output_image = clCreateImage2D(context, CL_MEM_READ_WRITE, &output_format, width, height, 0, NULL, &err);
    if(err < 0) {
        perror("occlustion_fill: Couldn't create the input image object");
        exit(1);
    };

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    if(err < 0) {
        perror("occlustion_fill, Error: clSetKernelArg, inputImage");
        exit(1);
    }

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
    if(err < 0) {
        perror("occlustion_fill, Error: clSetKernelArg, outputImage");
        exit(1);
    }

    size_t localMemSize = sizeof(float) * 4 * (1024 + 4 - 1) * (768 + 4 - 1);
    err = clSetKernelArg(kernel, 2, localMemSize, NULL);
    if(err < 0) {
        perror("occlustion_fill, Error: clSetKernelArg, localMemSize");
        exit(1);
    }

    // Execute the OpenCL kernel
    size_t globalWorkSize[2] = { width, height };
    size_t localWorkSize[2] = {16, 16};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &occlustion_fill_event);
    if(err < 0) {
        perror("occlustion_fill, Error: clEnqueueNDRangeKernel");
        exit(1);
    }

    printf("HELLO\n");
    // Read the output image back to the host
    err = clEnqueueReadImage(queue, output_image, CL_TRUE, (size_t[3]){0, 0, 0}, (size_t[3]){width, height, 1},
                             0, 0, (void*)output_im0 -> image, 0, NULL, &occlustion_fill_read_event);
    if(err < 0) {
        perror("occlustion_fill, Error: clEnqueueReadImage");
        exit(1);
    }

    clFinish(queue);

    output_im0 -> width = width;
    output_im0 -> height = height;

    time_to_occlustion_fill = getExecutionTime(occlustion_fill_event);
    read_time = getExecutionTime(occlustion_fill_read_event);

    clReleaseEvent(occlustion_fill_read_event);
    clReleaseEvent(occlustion_fill_event);

    printf("Time taken to do the occlustion_fill = %llu ns\n", time_to_occlustion_fill);
    printf("Time taken to read the output image (occlustion_fill) = %llu ns\n", read_time);
}

void openclFlowEx5(void) {
    printf("OpenCL Flow Example 5 STARTED\n");
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_int err;

    cl_kernel *kernels, kernel_resize_image, kernel_color_to_gray, kernel_zncc_left, kernel_zncc_right, kernel_cross_check, kernel_occlusion_fill;
    char kernel_name[20];
    cl_uint i, num_kernels;

    /* Open input file and read image data */
    Image *im0 = readImage(INPUT_FILE_0);
    Image *im1 = readImage(INPUT_FILE_1);
    const size_t width = im0 -> width;
    const size_t height = im0 -> height;
    const size_t new_width = width / 4;
    const size_t new_height = height / 4;

    Image *output_1_resized_im0 = createEmptyImage(new_width, new_height);
    Image *output_1_bw_im0 = createEmptyImage(new_width, new_height);
    Image *output_left_disparity_im0 = createEmptyImage(new_width, new_height);

    Image *output_2_resized_im0 = createEmptyImage(new_width, new_height);
    Image *output_2_bw_im0 = createEmptyImage(new_width, new_height);
    Image *output_right_disparity_im0 = createEmptyImage(new_width, new_height);
    Image *left_crosscheck_im0 = createEmptyImage(new_width, new_height);
    Image *output_left_occlusion_im0 = createEmptyImage(new_width, new_height);

    device = create_device();

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0) {
        perror("Error: clCreateContext");
        exit(1);
    }

    program = build_program(context, device, PROGRAM_FILE);

    /* Find out how many kernels are in the source file */
    err = clCreateKernelsInProgram(program, 0, NULL, &num_kernels);
    if(err < 0) {
        perror("Couldn't find any kernels");
        exit(1);
    }

    printf("Number of kernels: %u\n", num_kernels);

    /* Create a kernel for each function */
    kernels = (cl_kernel*) malloc(num_kernels * sizeof(cl_kernel));
    clCreateKernelsInProgram(program, num_kernels, kernels, NULL);

    // /* Search for the named kernel */
    for(i=0; i<num_kernels; i++) {
        clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME,
                        sizeof(kernel_name), kernel_name, NULL);
        if(strcmp(kernel_name, KERNEL_RESIZE_IMAGE) == 0) {
            kernel_resize_image = kernels[i];
            printf("Found resize_image kernel at index %u.\n", i);
        } else if(strcmp(kernel_name, KERNEL_COLOR_TO_GRAY) == 0) {
            kernel_color_to_gray = kernels[i];
            printf("Found color_to_gray kernel at index %u.\n", i);
        } else if(strcmp(kernel_name, KERNEL_ZNCC_LEFT) == 0) {
            kernel_zncc_left = kernels[i];
            printf("Found zncc_left kernel at index %u.\n", i);
        } else if(strcmp(kernel_name, KERNEL_ZNCC_RIGHT) == 0) {
            kernel_zncc_right = kernels[i];
            printf("Found zncc_right kernel at index %u.\n", i);
        } else if(strcmp(kernel_name, KERNEL_CROSS_CHECK) == 0) {
            kernel_cross_check = kernels[i];
            printf("Found cross_check kernel at index %u.\n", i);
        } else if(strcmp(kernel_name, KERNEL_OCCLUSION_FILL) == 0) {
            kernel_occlusion_fill = kernels[i];
            printf("Found occlustion kernel at index %u.\n", i);
        }
    }
    // cl_command_queue_properties props[3] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    // queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if(err < 0) {
        perror("Error: clCreateCommandQueue");
        exit(1);
    }

    /* Resize image size */
    resize_image(context, kernel_resize_image, queue, im0, output_1_resized_im0);
    /* Convert color image to gray scale image */
    convert_image_to_gray(context, kernel_color_to_gray, queue, output_1_resized_im0, output_1_bw_im0);

    /* Resize image size */
    resize_image(context, kernel_resize_image, queue, im1, output_2_resized_im0);
    /* Convert color image to gray scale image */
    convert_image_to_gray(context, kernel_color_to_gray, queue, output_2_resized_im0, output_2_bw_im0);

    /* Apply zncc kernel */
    apply_zncc(device, context, kernel_zncc_left, queue, output_1_bw_im0, output_2_bw_im0, output_left_disparity_im0);

    /* Apply zncc kernel */
    apply_zncc(device, context, kernel_zncc_right, queue, output_2_bw_im0, output_1_bw_im0, output_right_disparity_im0);

    /* Apply left crosscheck kernel */
    apply_crosscheck(context, kernel_cross_check, queue, output_left_disparity_im0, output_right_disparity_im0, left_crosscheck_im0);

    /* Apply left occlustion fill kernel */
    apply_occlusion_fill(context, kernel_occlusion_fill, queue, left_crosscheck_im0, output_left_occlusion_im0);

    saveImage(OUTPUT_1_RESIZE_OPENCL_FILE, output_1_resized_im0);
    saveImage(OUTPUT_1_BW_OPENCL_FILE, output_1_bw_im0);
    saveImage(OUTPUT_1_LEFT_DISPARITY_OPENCL_FILE, output_left_disparity_im0);

    saveImage(OUTPUT_2_RESIZE_OPENCL_FILE, output_2_resized_im0);
    saveImage(OUTPUT_2_BW_OPENCL_FILE, output_2_bw_im0);
    saveImage(OUTPUT_2_LEFT_DISPARITY_OPENCL_FILE, output_right_disparity_im0);

    saveImage(OUTPUT_3_CROSSCHECK_LEFT_OPENCL_FILE, left_crosscheck_im0);

    saveImage(OUTPUT_4_OCCLUSTION_LEFT_OPENCL_FILE, output_left_occlusion_im0);

    /* Deallocate resources */
    freeImage(im0);
    freeImage(im1);
    freeImage(output_1_resized_im0);
    freeImage(output_1_bw_im0);
    freeImage(output_left_disparity_im0);
    freeImage(output_2_resized_im0);
    freeImage(output_2_bw_im0);
    freeImage(output_right_disparity_im0);
    clReleaseKernel(kernel_color_to_gray);
    clReleaseKernel(kernel_resize_image);
    clReleaseKernel(kernel_zncc_left);
    clReleaseKernel(kernel_zncc_right);
    clReleaseKernel(kernel_cross_check);
    clReleaseKernel(kernel_occlusion_fill);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    printf("OpenCL Flow Example 5 ENDED\n");
}
