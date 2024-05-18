__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__constant float gassian_kernel[5][5] = {
{0.0030, 0.0133, 0.0219, 0.0133, 0.0030},
{0.0133, 0.0596, 0.0983, 0.0596, 0.0133},
{0.0219, 0.0983, 0.1621, 0.0983, 0.0219},
{0.0133, 0.0596, 0.0983, 0.0596, 0.0133},
{0.0030, 0.0133, 0.0219, 0.0133, 0.0030}
};

__kernel void resize_image(__read_only image2d_t inputImage, __write_only image2d_t outputImage) {
    const int2 output_cord = (int2)(get_global_id(0), get_global_id(1));

    // Read the color pixel from the input image
    int2 input_image_cord = (int2)(output_cord.x * 4, output_cord.y * 4);
    float4 colorPixel = read_imagef(inputImage, sampler, input_image_cord);

    // Write the value to the output image
    write_imagef(outputImage, output_cord, colorPixel);
}

__kernel void color_to_gray(__read_only image2d_t inputImage, __write_only image2d_t outputImage) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));

    // Read the color pixel from the input image
    float4 colorPixel = read_imagef(inputImage, sampler, pos);

    // Convert RGB to grayscale using luminance (Y = 0.2126*R + 0.7152*G + 0.0722*B)
    float grayscaleValue = 0.2126f * colorPixel.x + 0.7152f * colorPixel.y + 0.0722f * colorPixel.z;

    // Write the grayscale value to the output image
    write_imagef(outputImage, pos, (float4)(grayscaleValue, grayscaleValue, grayscaleValue, 1.0f));
}

#define WINDOW_SIZE 15
#define WINDOW_HALF_SIZE 7

__kernel void left_disparity(__read_only image2d_t inputImage1, __read_only image2d_t inputImage2, __write_only image2d_t outputImage) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));

    float sum = 0.0f;

    float image1Window[WINDOW_SIZE * WINDOW_SIZE];
    int2 offsetPos;
    float4 color;
    int flatIndex;
    for (int i = -WINDOW_HALF_SIZE; i <= WINDOW_HALF_SIZE; ++i) {
        for (int j = -WINDOW_HALF_SIZE; j <= WINDOW_HALF_SIZE; ++j) {
            flatIndex = (i + WINDOW_HALF_SIZE) * WINDOW_SIZE + (j + WINDOW_HALF_SIZE);
            offsetPos = pos + (int2)(i, j);
            color = read_imagef(inputImage1, sampler, offsetPos);
            image1Window[flatIndex] = color.x;
            sum += color.x;
        }
    }
    const float image1Mean = sum / (WINDOW_SIZE * WINDOW_SIZE);

    const int MAX_DISP = 65;

    float bestDisp = 0.0f;
    float max_zncc_x = 0.0f;
    float sum2;
    float zncc;
    float normalizedDisp;
    float4 result;
    float firstDiff;
    float secondDiff;
    for (int d = 0; d < MAX_DISP; ++d) {

        sum2 = 0.0f;
        float image2Window[WINDOW_SIZE * WINDOW_SIZE];

        for (int i = -WINDOW_HALF_SIZE; i <= WINDOW_HALF_SIZE; ++i) {
            for (int j = -WINDOW_HALF_SIZE; j <= WINDOW_HALF_SIZE; ++j) {

                flatIndex = (i + WINDOW_HALF_SIZE) * WINDOW_SIZE + (j + WINDOW_HALF_SIZE);
                offsetPos = pos + (int2)(i - d, j);
                color = read_imagef(inputImage2, sampler, offsetPos);
                image2Window[flatIndex] = color.x;
                sum2 += color.x;
            }
        }
        const float image2Mean = sum2 / (WINDOW_SIZE * WINDOW_SIZE);

        float diffMultiSum = 0.0f;
        float squaredSum2 = 0.0f;
        float squaredSum1 = 0.0f;

        // TODO: check if loop unrolling helps
        for (int i = 0; i < WINDOW_SIZE * WINDOW_SIZE; ++i) {
            firstDiff = image1Window[i] - image1Mean;
            secondDiff = image2Window[i] - image2Mean;
            diffMultiSum += firstDiff * secondDiff;

            firstDiff = firstDiff * firstDiff;
            secondDiff = secondDiff * secondDiff;

            squaredSum1 += firstDiff;
            squaredSum2 += secondDiff;
        }

        zncc = diffMultiSum / (sqrt(squaredSum1) * sqrt(squaredSum2));
        if (zncc > max_zncc_x) {
            bestDisp = abs(d);
            max_zncc_x = zncc;
        }
    }

    normalizedDisp = (bestDisp / MAX_DISP);
   // printf("normalizedDisp: %f\n", normalizedDisp);

    result = (float4){normalizedDisp, normalizedDisp, normalizedDisp, 1.0f};
    write_imagef(outputImage, pos, result);
}

__kernel void right_disparity(__read_only image2d_t inputImage1, __read_only image2d_t inputImage2, __write_only image2d_t outputImage) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));

    float sum = 0.0f;

    float image1Window[WINDOW_SIZE * WINDOW_SIZE];
    int2 offsetPos;
    float4 color;

    for (int i = -WINDOW_HALF_SIZE; i <= WINDOW_HALF_SIZE; ++i) {
        for (int j = -WINDOW_HALF_SIZE; j <= WINDOW_HALF_SIZE; ++j) {
            int flatIndex = (i + WINDOW_HALF_SIZE) * WINDOW_SIZE + (j + WINDOW_HALF_SIZE);    
            offsetPos = pos + (int2)(i, j);
            color = read_imagef(inputImage1, sampler, offsetPos);
            image1Window[flatIndex] = color.x;
            sum += color.x;
        }
    }
    const float image1Mean = sum / (WINDOW_SIZE * WINDOW_SIZE);

    const int MAX_DISP = 65;

    float bestDisp = 0.0f;
    float max_zncc_x = 0.0f;
    float zncc;
    float sum2;
    float normalizedDisp;
    float4 result;
    float firstDiff;
    float secondDiff;
    for (int d = 0; d < MAX_DISP; ++d) {

        sum2 = 0.0f;
        float image2Window[WINDOW_SIZE * WINDOW_SIZE];

        for (int i = -WINDOW_HALF_SIZE; i <= WINDOW_HALF_SIZE; ++i) {
            for (int j = -WINDOW_HALF_SIZE; j <= WINDOW_HALF_SIZE; ++j) {

                int flatIndex2 = (i + WINDOW_HALF_SIZE) * WINDOW_SIZE + (j + WINDOW_HALF_SIZE);   
                offsetPos = pos + (int2)(i + d, j);
                color = read_imagef(inputImage2, sampler, offsetPos);
                image2Window[flatIndex2] = color.x;
                sum2 += color.x;
            }
        }
        const float image2Mean = sum2 / (WINDOW_SIZE * WINDOW_SIZE);

        float diffMultiSum = 0.0f;
        float squaredSum2 = 0.0f;
        float squaredSum1 = 0.0f;

        // TODO: check if loop unrolling helps
        for (int i = 0; i < WINDOW_SIZE * WINDOW_SIZE; ++i) {

            firstDiff = image1Window[i] - image1Mean;
            secondDiff = image2Window[i] - image2Mean;
            diffMultiSum += firstDiff * secondDiff;

            firstDiff = firstDiff * firstDiff;
            secondDiff = secondDiff * secondDiff;
            squaredSum1 += firstDiff;
            squaredSum2 += secondDiff;
        }

        zncc = diffMultiSum / (sqrt(squaredSum1) * sqrt(squaredSum2));
        if (zncc > max_zncc_x) {
            bestDisp = abs(d);
            max_zncc_x = zncc;
        }
    }

    normalizedDisp = (bestDisp / MAX_DISP);

    result = (float4){normalizedDisp, normalizedDisp, normalizedDisp, 1.0f};
    write_imagef(outputImage, pos, result);
}

__kernel void crosscheck(__read_only image2d_t inputImage1, __read_only image2d_t inputImage2, __write_only image2d_t outputImage) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));

    const float4 image1Pixel = read_imagef(inputImage1, sampler, pos);
    const float4 image2Pixel = read_imagef(inputImage2, sampler, pos);

    float4 crosscheck_output = (float4)(0.0f, 0.0f, 0.0f, 1.0f);

    if (fabs(image1Pixel.x - image2Pixel.x) <= 0.588f) {
        crosscheck_output.x = image1Pixel.x;
        crosscheck_output.y = image1Pixel.y;
        crosscheck_output.z = image1Pixel.z;
    }

    write_imagef(outputImage, pos, crosscheck_output);
}


__kernel void occlusion_fill(__read_only image2d_t inputImage, __write_only image2d_t outputImage) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));

    float neighbor_pixels[5][5];

    const float4 imagePixel = read_imagef(inputImage, sampler, pos);
    float4 occlusion_output = (float4)(0.0f, 0.0f, 0.0f, 1.0f);

        int2 offsetPos;
        float prevNonZeroValue = 0.0f;
        int localXIndex;
        int localYIndex;
        float sum;
        float4 color;

    if (imagePixel.x < 10E-3 && imagePixel.y < 10E-3 && imagePixel.z < 10E-3 ) {

        sum = 0.0f;

        for (int i = -2; i <= 2; ++i) {
            localYIndex = (i + 2) * 5;
            for (int j = -2; j <= 2; ++j) {
                localXIndex = (j + 2);

                offsetPos = pos + (int2)(i, j);
                color = read_imagef(inputImage, sampler, offsetPos);
                neighbor_pixels[localYIndex][localXIndex] = color.x;
                if (color.x != 0) {
                    prevNonZeroValue = color.x;
                    break;
                }
            }
        }

        float weight;
        float fColor;
        for (int i = -2; i <= 2; ++i) {
            localYIndex = (i + 2) * 5;
            for (int j = -2; j <= 2; ++j) {
                localXIndex = (j + 2);

                fColor = neighbor_pixels[localYIndex][localXIndex];
                if (fColor != 0) {
                    prevNonZeroValue = fColor;
                } else {
                    fColor = prevNonZeroValue;
                }

                float weight = gassian_kernel[i + 2][j + 2];
                sum += weight * fColor;
            }
        }
        occlusion_output.x = sum;
        occlusion_output.y = sum;
        occlusion_output.z = sum;

    } else {
        occlusion_output.x = imagePixel.x;
        occlusion_output.y = imagePixel.y;
        occlusion_output.z = imagePixel.z;
    }

    write_imagef(outputImage, pos, occlusion_output);
}
