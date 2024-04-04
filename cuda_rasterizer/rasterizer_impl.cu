/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CPU_RASTERIZER_H_INCLUDED
#define CPU_RASTERIZER_H_INCLUDED

#include <cstdint>
#include <vector>
#include <functional>
#include <glm/glm.hpp>

namespace CpuRasterizer {

    // Helper function to find the next-highest bit of the MSB
    uint32_t getHigherMsb(uint32_t n) {
        uint32_t msb = sizeof(n) * 4;
        uint32_t step = msb;
        while (step > 1) {
            step /= 2;
            if (n >> msb)
                msb += step;
            else
                msb -= step;
        }
        if (n >> msb)
            msb++;
        return msb;
    }

    // Function to mark Gaussians as visible/invisible based on view frustum testing
    void markVisible(
        int P,
        float* means3D,
        float* viewmatrix,
        float* projmatrix,
        bool* present);

    // Function to duplicate Gaussian instances with keys for sorting
    void duplicateWithKeys(
        int P,
        const float2* points_xy,
        const float* depths,
        const uint32_t* offsets,
        uint64_t* gaussian_keys_unsorted,
        uint32_t* gaussian_values_unsorted,
        int* radii,
        dim3 grid);

    // Function to identify tile ranges in sorted list of Gaussian keys
    void identifyTileRanges(
        int L,
        uint64_t* point_list_keys,
        uint2* ranges);

    // Forward rendering procedure for differentiable rasterization of Gaussians
    int forward(
        std::function<char* (size_t)> geometryBuffer,
        std::function<char* (size_t)> binningBuffer,
        std::function<char* (size_t)> imageBuffer,
        const int P, int D, int M,
        const float* background,
        const int width, int height,
        const float* means3D,
        const float* shs,
        const float* colors_precomp,
        const float* opacities,
        const float* scales,
        const float scale_modifier,
        const float* rotations,
        const float* cov3D_precomp,
        const float* viewmatrix,
        const float* projmatrix,
        const float* cam_pos,
        const float tan_fovx, float tan_fovy,
        const bool prefiltered,
        float* out_color,
        int* radii,
        bool debug);

    // Produce necessary gradients for optimization, corresponding to forward render pass
    void backward(
        const int P, int D, int M, int R,
        const float* background,
        const int width, int height,
        const float* means3D,
        const float* shs,
        const float* colors_precomp,
        const float* scales,
        const float scale_modifier,
        const float* rotations,
        const float* cov3D_precomp,
        const float* viewmatrix,
        const float* projmatrix,
        const float* campos,
        const float tan_fovx, float tan_fovy,
        const int* radii,
        char* geom_buffer,
        char* binning_buffer,
        char* img_buffer,
        const float* dL_dpix,
        float* dL_dmean2D,
        float* dL_dconic,
        float* dL_dopacity,
        float* dL_dcolor,
        float* dL_dmean3D,
        float* dL_dcov3D,
        float* dL_dsh,
        float* dL_dscale,
        float* dL_drot,
        bool debug);
}

#endif // CPU_RASTERIZER_H_INCLUDED
