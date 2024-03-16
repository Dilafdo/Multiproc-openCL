#include "pngloader.h"
#include <stdlib.h>

Image* CrossCheck(Image * image1, Image* image2, int threshold) {
    Image * crossCheckedImage = createEmptyImage(image1->width, image1->height);

    int height = image1->height;
    int width = image1->width;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            size_t index = 4 * i * width + 4 * j;
            crossCheckedImage->image[index + 3] = image1->image[index + 3]; // alpha channel
            if (abs(image1->image[index] - image2->image[index]) > threshold) {
                crossCheckedImage->image[index] = 0;
                crossCheckedImage->image[index + 1] = 0;
                crossCheckedImage->image[index + 2] = 0;
            } else {
                crossCheckedImage->image[index] = image1->image[index];
                crossCheckedImage->image[index + 1] = image1->image[index + 1];
                crossCheckedImage->image[index + 2] = image1->image[index + 2];
            }
        }
    }
    return crossCheckedImage;
}

Image* CrossCheck_MP(Image * image1, Image* image2, int threshold) {
    Image * crossCheckedImage = createEmptyImage(image1->width, image1->height);

    int height = image1->height;
    int width = image1->width;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            size_t index = 4 * i * width + 4 * j;
            crossCheckedImage->image[index + 3] = image1->image[index + 3]; // alpha channel
            if (abs(image1->image[index] - image2->image[index]) > threshold) {
                crossCheckedImage->image[index] = 0;
                crossCheckedImage->image[index + 1] = 0;
                crossCheckedImage->image[index + 2] = 0;
            } else {
                crossCheckedImage->image[index] = image1->image[index];
                crossCheckedImage->image[index + 1] = image1->image[index + 1];
                crossCheckedImage->image[index + 2] = image1->image[index + 2];
            }
        }
    }
    return crossCheckedImage;
}