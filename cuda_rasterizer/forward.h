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

#ifndef CPU_RASTERIZER_FORWARD_H_INCLUDED
#define CPU_RASTERIZER_FORWARD_H_INCLUDED

#include <vector>
#include <glm/glm.hpp>

namespace FORWARD
{
    // Perform initial steps for each Gaussian prior to rasterization.
    void preprocess(int P, int D, int M,
                    const std::vector<float>& orig_points,
                    const std::vector<glm::vec3>& scales,
                    float scale_modifier,
                    const std::vector<glm::vec4>& rotations,
                    const std::vector<float>& opacities,
                    const std::vector<float>& shs,
                    std::vector<bool>& clamped,
                    const std::vector<float>& cov3D_precomp,
                    const std::vector<float>& colors_precomp,
                    const std::vector<float>& viewmatrix,
                    const std::vector<float>& projmatrix,
                    const std::vector<glm::vec3>& cam_pos,
                    int W, int H,
                    float focal_x, float focal_y,
                    float tan_fovx, float tan_fovy,
                    std::vector<int>& radii,
                    std::vector<float2>& points_xy_image,
                    std::vector<float>& depths,
                    std::vector<float>& cov3Ds,
                    std::vector<float>& colors,
                    std::vector<float4>& conic_opacity,
                    const dim3 grid,
                    std::vector<uint32_t>& tiles_touched,
                    bool prefiltered);

    // Main rasterization method.
    void render(
        const dim3 grid, dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int W, int H,
        const float2* points_xy_image,
        const float* features,
        const float4* conic_opacity,
        float* final_T,
        uint32_t* n_contrib,
        const float* bg_color,
        float* out_color);
}

#endif
