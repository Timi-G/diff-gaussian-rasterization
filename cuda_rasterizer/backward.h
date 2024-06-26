#ifndef CPU_RASTERIZER_BACKWARD_H_INCLUDED
#define CPU_RASTERIZER_BACKWARD_H_INCLUDED

#include <glm/glm.hpp>

namespace BACKWARD
{
    void render(
        const uint2* ranges,
        const uint32_t* point_list,
        int W, int H,
        const float* bg_color,
        const float2* means2D,
        const float4* conic_opacity,
        const float* colors,
        const float* final_Ts,
        const uint32_t* n_contrib,
        const float* dL_dpixels,
        float3* dL_dmean2D,
        float4* dL_dconic2D,
        float* dL_dopacity,
        float* dL_dcolors);

    void preprocess(
        int P, int D, int M,
        const float3* means,
        const int* radii,
        const float* shs,
        const bool* clamped,
        const glm::vec3* scales,
        const glm::vec4* rotations,
        const float scale_modifier,
        const float* cov3Ds,
        const float* view,
        const float* proj,
        const float focal_x, float focal_y,
        const float tan_fovx, float tan_fovy,
        const glm::vec3* campos,
        const float3* dL_dmean2D,
        const float* dL_dconics,
        glm::vec3* dL_dmeans,
        float* dL_dcolor,
        float* dL_dcov3D,
        float* dL_dsh,
        glm::vec3* dL_dscale,
        glm::vec4* dL_drot);
}

#endif
