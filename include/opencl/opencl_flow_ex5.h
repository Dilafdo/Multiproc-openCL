//
// Created by ruksh on 02/03/2024.
//

#ifndef MULTIPROCOPENCL_OPENCL_FLOW_EX5_H
#define MULTIPROCOPENCL_OPENCL_FLOW_EX5_H
#include <opencl_include.h>

void openclFlowEx5(void);
cl_device_id create_device(void);
cl_program build_program(cl_context ctx, cl_device_id device, const char* filename);
cl_ulong getExecutionTime(cl_event event);
void resize_image(cl_context context, cl_kernel kernel, cl_command_queue queue, const Image *im0, Image *output_im0);
void convert_image_to_gray(cl_context context, cl_kernel kernel, cl_command_queue queue, const Image *im0, Image *output_im0);
void apply_zncc(cl_device_id device, cl_context context, cl_kernel kernel, cl_command_queue queue, const Image *im0, const Image *im1, Image *output_im0);
void apply_crosscheck(cl_context context, cl_kernel kernel, cl_command_queue queue, const Image *im0, const Image *im1, Image *output_im0);
void apply_occlusion_fill(cl_device_id device, cl_context context, cl_kernel kernel, cl_command_queue queue, const Image *im0, Image *output_im0);

#endif //MULTIPROCOPENCL_OPENCL_FLOW_EX5_H
