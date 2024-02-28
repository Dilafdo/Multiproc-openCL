//
// Created by ruksh on 28/02/2024.
//

#ifndef MULTIPROCOPENCL_DRIVER_H
#define MULTIPROCOPENCL_DRIVER_H

void runZnccFlowForOneImage(const char * imagePath, const char * outputPath);
void zncc_flow_driver(const char *imagePath, const char *outputPath);
void createSampleTestPng();

#endif //MULTIPROCOPENCL_DRIVER_H