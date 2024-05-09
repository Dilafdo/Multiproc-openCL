__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

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
    int x = get_global_id(0);
    int y = get_global_id(1);

    const int2 pos = (int2)(x, y);

    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    float4 image1Window[WINDOW_SIZE * WINDOW_SIZE];

    for (int i = -WINDOW_HALF_SIZE; i <= WINDOW_HALF_SIZE; ++i) {
        for (int j = -WINDOW_HALF_SIZE; j <= WINDOW_HALF_SIZE; ++j) {
            int flatIndex = (i + WINDOW_HALF_SIZE) * WINDOW_SIZE + (j + WINDOW_HALF_SIZE);    
            const int2 offsetPos = pos + (int2)(i, j);
            const float4 color = read_imagef(inputImage1, sampler, offsetPos);
            image1Window[flatIndex] = color;
            sum += color;
        }
    }
    const float4 image1Mean = sum / (WINDOW_SIZE * WINDOW_SIZE);

    const int MAX_DISP = 65;

    float bestDisp = 0.0f;
    float4 max_zncc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int d = 0; d < MAX_DISP; ++d) {

        float4 sum2 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float4 image2Window[WINDOW_SIZE * WINDOW_SIZE];

        for (int i = -WINDOW_HALF_SIZE; i <= WINDOW_HALF_SIZE; ++i) {
            for (int j = -WINDOW_HALF_SIZE; j <= WINDOW_HALF_SIZE; ++j) {

                int flatIndex2 = (i + WINDOW_HALF_SIZE) * WINDOW_SIZE + (j + WINDOW_HALF_SIZE);   
                const int2 offsetPosImage2 = pos + (int2)(i - d, j);
                const float4 colorIm2 = read_imagef(inputImage2, sampler, offsetPosImage2);
                image2Window[flatIndex2] = colorIm2;
                sum2 += colorIm2;
            }
        }
        const float4 image2Mean = sum2 / (WINDOW_SIZE * WINDOW_SIZE);

        float4 diffMultiSum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float4 squaredSum2 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float4 squaredSum1 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

        for (int i = 0; i < WINDOW_SIZE * WINDOW_SIZE; ++i) {
            const float4 firstDiff = image1Window[i] - image1Mean;
            const float4 firstDiffSquared = firstDiff * firstDiff;
            const float4 secondDiff = image2Window[i] - image2Mean;
            const float4 secondDiffSquared = secondDiff * secondDiff;
            const float4 diffMultiplied = firstDiff * secondDiff;
            diffMultiSum += diffMultiplied;
            squaredSum1 += firstDiffSquared;
            squaredSum2 += secondDiffSquared;
        }

        float4 zncc = diffMultiSum / (sqrt(squaredSum1) * sqrt(squaredSum2));
        if (zncc.x > max_zncc.x) {
            bestDisp = abs(d);
            max_zncc = zncc;
        }
    }

    const float normalizedDisp = (bestDisp / MAX_DISP);
   // printf("normalizedDisp: %f\n", normalizedDisp);

    float4 result = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    result.x = normalizedDisp;
    result.y = normalizedDisp;
    result.z = normalizedDisp;
    result.w = 1.0f;
    write_imagef(outputImage, pos, result);
}

__kernel void right_disparity(__read_only image2d_t inputImage1, __read_only image2d_t inputImage2, __write_only image2d_t outputImage) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));

    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    float4 image1Window[WINDOW_SIZE * WINDOW_SIZE];

    for (int i = -WINDOW_HALF_SIZE; i <= WINDOW_HALF_SIZE; ++i) {
        for (int j = -WINDOW_HALF_SIZE; j <= WINDOW_HALF_SIZE; ++j) {
            int flatIndex = (i + WINDOW_HALF_SIZE) * WINDOW_SIZE + (j + WINDOW_HALF_SIZE);    
            const int2 offsetPos = pos + (int2)(i, j);
            const float4 color = read_imagef(inputImage1, sampler, offsetPos);
            image1Window[flatIndex] = color;
            sum += color;
        }
    }
    const float4 image1Mean = sum / (WINDOW_SIZE * WINDOW_SIZE);

    const int MAX_DISP = 65;

    float bestDisp = 0.0f;
    float4 max_zncc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int d = 0; d < MAX_DISP; ++d) {

        float4 sum2 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float4 image2Window[WINDOW_SIZE * WINDOW_SIZE];

        for (int i = -WINDOW_HALF_SIZE; i <= WINDOW_HALF_SIZE; ++i) {
            for (int j = -WINDOW_HALF_SIZE; j <= WINDOW_HALF_SIZE; ++j) {

                int flatIndex2 = (i + WINDOW_HALF_SIZE) * WINDOW_SIZE + (j + WINDOW_HALF_SIZE);   
                const int2 offsetPosImage2 = pos + (int2)(i + d, j);
                const float4 colorIm2 = read_imagef(inputImage2, sampler, offsetPosImage2);
                image2Window[flatIndex2] = colorIm2;
                sum2 += colorIm2;
            }
        }
        const float4 image2Mean = sum2 / (WINDOW_SIZE * WINDOW_SIZE);

        float4 diffMultiSum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float4 squaredSum2 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float4 squaredSum1 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

        for (int i = 0; i < WINDOW_SIZE * WINDOW_SIZE; ++i) {
            const float4 firstDiff = image1Window[i] - image1Mean;
            const float4 firstDiffSquared = firstDiff * firstDiff;
            const float4 secondDiff = image2Window[i] - image2Mean;
            const float4 secondDiffSquared = secondDiff * secondDiff;
            const float4 diffMultiplied = firstDiff * secondDiff;
            diffMultiSum += diffMultiplied;
            squaredSum1 += firstDiffSquared;
            squaredSum2 += secondDiffSquared;
        }

        float4 zncc = diffMultiSum / (sqrt(squaredSum1) * sqrt(squaredSum2));
        if (zncc.x > max_zncc.x) {
            bestDisp = abs(d);
            max_zncc = zncc;
        }
    }

    const float normalizedDisp = (bestDisp / MAX_DISP);
   // printf("normalizedDisp: %f\n", normalizedDisp);

    float4 result = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    result.x = normalizedDisp;
    result.y = normalizedDisp;
    result.z = normalizedDisp;
    result.w = 1.0f;
    write_imagef(outputImage, pos, result);
}

__kernel void crosscheck(__read_only image2d_t inputImage1, __read_only image2d_t inputImage2, __write_only image2d_t outputImage) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));

    const float threshold = 0.588f;

    const float4 image1Pixel = read_imagef(inputImage1, sampler, pos);
    const float4 image2Pixel = read_imagef(inputImage2, sampler, pos);

    float4 crosscheck_output = (float4)(0.0f, 0.0f, 0.0f, 1.0f);

    if (fabs(image1Pixel.x - image2Pixel.x) <= threshold) {
        crosscheck_output.x = image1Pixel.x;
        crosscheck_output.y = image1Pixel.y;
        crosscheck_output.z = image1Pixel.z;
    }

    write_imagef(outputImage, pos, crosscheck_output);
}

__kernel void occlusion_fill(__read_only image2d_t inputImage, __write_only image2d_t outputImage) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));

    const float4 imagePixel = read_imagef(inputImage, sampler, pos);
    float4 occlusion_output = (float4)(0.0f, 0.0f, 0.0f, 1.0f);

    const float gassian_kernel[5][5] = {
        {0.0030, 0.0133, 0.0219, 0.0133, 0.0030},
        {0.0133, 0.0596, 0.0983, 0.0596, 0.0133},
        {0.0219, 0.0983, 0.1621, 0.0983, 0.0219},
        {0.0133, 0.0596, 0.0983, 0.0596, 0.0133},
        {0.0030, 0.0133, 0.0219, 0.0133, 0.0030}
    };

    if (imagePixel.x == 0 && imagePixel.y == 0 && imagePixel.z == 0) {

        float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float prevNonZeroValue = 0.0f;

        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {
                const int2 offsetPos = pos + (int2)(i, j);
                float4 color = read_imagef(inputImage, sampler, offsetPos);
                if (color.x != 0) {
                    prevNonZeroValue = color.x;
                    break;
                }
            }
        }

        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {

                const int2 offsetPos = pos + (int2)(i, j);
                float4 color = read_imagef(inputImage, sampler, offsetPos);
                if (color.x != 0) {
                    prevNonZeroValue = color.x;
                } else {
                    color.x = prevNonZeroValue;
                    color.y = prevNonZeroValue;
                    color.z = prevNonZeroValue;
                }

                const float weight = gassian_kernel[i + 2][j + 2];
                sum += weight * color;
            }
        }
        occlusion_output.x = sum.x;
        occlusion_output.y = sum.y;
        occlusion_output.z = sum.z;

    } else {
        occlusion_output.x = imagePixel.x;
        occlusion_output.y = imagePixel.y;
        occlusion_output.z = imagePixel.z;
    }

    write_imagef(outputImage, pos, occlusion_output);
}