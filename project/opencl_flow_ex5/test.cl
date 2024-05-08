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

__kernel void occlusion_fill(__read_only image2d_t inputImage, __write_only image2d_t outputImage, __local float4* localMem) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 localPos = (int2)(get_local_id(0), get_local_id(1));
    const int2 groupSize = (int2)(get_local_size(0), get_local_size(1));

    // Define the size of the local memory tile including the Gaussian kernel radius
    const int radius = 2;
    const int diameter = radius * 2 + 1;
    const int localWidth = groupSize.x + diameter - 1;
    const int localHeight = groupSize.y + diameter - 1;
    const int2 localDim = (int2)(localWidth, localHeight);

    // Load data into local memory
    const int2 localIndex = (int2)(localPos.x + radius, localPos.y + radius);
    if ((localIndex.x < localDim.x) && (localIndex.y < localDim.y)) {
        const int2 globalIndex = pos - (int2)(radius, radius) + (int2)(get_local_id(0), get_local_id(1));
        if (globalIndex.x >= 0 && globalIndex.x < get_image_width(inputImage) &&
            globalIndex.y >= 0 && globalIndex.y < get_image_height(inputImage)) {
            localMem[localIndex.y * localWidth + localIndex.x] = read_imagef(inputImage, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, globalIndex);
        } else {
            localMem[localIndex.y * localWidth + localIndex.x] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    
    // Synchronize to ensure all local memory is populated
    barrier(CLK_LOCAL_MEM_FENCE);

    // Gaussian kernel data
    const float gaussian_kernel[5][5] = {
        {0.0030, 0.0133, 0.0219, 0.0133, 0.0030},
        {0.0133, 0.0596, 0.0983, 0.0596, 0.0133},
        {0.0219, 0.0983, 0.1621, 0.0983, 0.0219},
        {0.0133, 0.0596, 0.0983, 0.0596, 0.0133},
        {0.0030, 0.0133, 0.0219, 0.0133, 0.0030}
    };

    // Perform computation using local memory
    float4 occlusion_output = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    if (read_imagef(inputImage, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, pos).x == 0.0f) {
        float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float prevNonZeroValue = 0.0f;

        for (int i = -radius; i <= radius; ++i) {
            for (int j = -radius; j <= radius; ++j) {
                const int2 localOffset = localIndex + (int2)(i, j);
                float4 color = localMem[localOffset.y * localWidth + localOffset.x];
                if (color.x != 0) {
                    prevNonZeroValue = color.x;
                } else {
                    color = (float4)(prevNonZeroValue, prevNonZeroValue, prevNonZeroValue, 1.0f);
                }

                const float weight = gaussian_kernel[i + radius][j + radius];
                sum += weight * color;
            }
        }

        occlusion_output = sum;
    } else {
        occlusion_output = read_imagef(inputImage, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, pos);
    }

    // Write output pixel
    write_imagef(outputImage, pos, occlusion_output);
}