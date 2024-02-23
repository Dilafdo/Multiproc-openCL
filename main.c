//
// Created by ruksh on 21/02/2024.
//

#include <stdio.h>
#include <stdlib.h>
#include <pngloader.h>

void createSampleTestPng() {
    const char* filename = "test.png";

    /*generate some image*/
    unsigned width = 512, height = 512;
    unsigned char* image = malloc(width * height * 4);
    unsigned x, y;
    for(y = 0; y < height; y++)
        for(x = 0; x < width; x++) {
            image[4 * width * y + 4 * x + 0] = 255 * !(x & y);
            image[4 * width * y + 4 * x + 1] = x ^ y;
            image[4 * width * y + 4 * x + 2] = x | y;
            image[4 * width * y + 4 * x + 3] = 255;
        }

    /*run an example*/
    encodeOneStep(filename, image, width, height);

    free(image);
}

int main() {
    printf("Starting Multiprocessor Programming project!\n");
    char* image0Name = "im0.png";
    char* image1Name = "im1.png";

    Image *im0 = loadImage(image0Name);
    Image *im1 = loadImage(image1Name);

    saveImage("im0_copy.png", im0);
    saveImage("im1_copy.png", im1);


    freeImage(im0);
    freeImage(im1);

    printf("Stopping Multiprocessor Programming project!\n");
    return 0;
}


