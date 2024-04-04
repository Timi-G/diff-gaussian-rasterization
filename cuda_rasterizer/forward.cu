#include <iostream>
#include <vector>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace cg = cooperative_groups; // Removed, not needed for CPU

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const std::vector<glm::vec3>& means, glm::vec3 campos,
                             const std::vector<float>& shs, std::vector<bool>& clamped)
{
    glm::vec3 pos = means[idx];
    glm::vec3 dir = pos - campos;
    dir = dir / glm::length(dir);

    glm::vec3 result = SH_C0 * shs[idx * max_coeffs];

    if (deg > 0)
    {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result = result - SH_C1 * y * shs[idx * max_coeffs + 1] + SH_C1 * z * shs[idx * max_coeffs + 2] -
                 SH_C1 * x * shs[idx * max_coeffs + 3];

        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                     SH_C2[0] * xy * shs[idx * max_coeffs + 4] +
                     SH_C2[1] * yz * shs[idx * max_coeffs + 5] +
                     SH_C2[2] * (2.0f * zz - xx - yy) * shs[idx * max_coeffs + 6] +
                     SH_C2[3] * xz * shs[idx * max_coeffs + 7] +
                     SH_C2[4] * (xx - yy) * shs[idx * max_coeffs + 8];

            if (deg > 2)
            {
                result = result +
                         SH_C3[0] * y * (3.0f * xx - yy) * shs[idx * max_coeffs + 9] +
                         SH_C3[1] * xy * z * shs[idx * max_coeffs + 10] +
                         SH_C3[2] * y * (4.0f * zz - xx - yy) * shs[idx * max_coeffs + 11] +
                         SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * shs[idx * max_coeffs + 12] +
                         SH_C3[4] * x * (4.0f * zz - xx - yy) * shs[idx * max_coeffs + 13] +
                         SH_C3[5] * z * (xx - yy) * shs[idx * max_coeffs + 14] +
                         SH_C3[6] * x * (xx - 3.0f * yy) * shs[idx * max_coeffs + 15];
            }
        }
    }
    result += 0.5f;

    // RGB colors are clamped to positive values. If values are
    // clamped, we need to keep track of this for the backward pass.
    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
    return glm::max(result, glm::vec3(0.0f));
}

// Forward version of 2D covariance matrix computation
glm::vec3 computeCov2D(const glm::vec3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy,
                       const std::vector<float>& cov3D, const std::vector<float>& viewmatrix)
{
    float3 t = transformPoint4x3(mean, viewmatrix);

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = std::min(limx, std::max(-limx, txtz)) * t.z;
    t.y = std::min(limy, std::max(-limy, tytz)) * t.z;

    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    glm::mat3 T = W * J;

    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return { cov[0][0], cov[0][1], cov[1][1] };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
void computeCov3D(const glm::vec3& scale, float mod, const glm::vec4& rot, std::vector<float>& cov3D)
{
    // Create scaling matrix
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    // Normalize quaternion to get valid rotation
    glm::vec4 q = rot;// / glm::length(rot);
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    // Compute rotation matrix from quaternion
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 M = S * R;

    // Compute 3D world covariance matrix Sigma
    glm::mat3 Sigma = glm::transpose(M) * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

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
                std::vector<float>& rgb,
                std::vector<float4>& conic_opacity,
                const dim3 grid,
                std::vector<uint32_t>& tiles_touched,
                bool prefiltered)
{
    for (int idx = 0; idx < P; ++idx)
    {
        // Initialize radius and touched tiles to 0. If this isn't changed,
        // this Gaussian will not be processed further.
        radii[idx] = 0;
        tiles_touched[idx] = 0;

        // Perform near culling, quit if outside.
        glm::vec3 p_view;
        if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
            continue;

        // Transform point by projecting
        glm::vec3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
        glm::vec4 p_hom = transformPoint4x4(p_orig, projmatrix);
        float p_w = 1.0f / (p_hom.w + 0.0000001f);
        glm::vec3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

        // If 3D covariance matrix is precomputed, use it, otherwise compute
        // from scaling and rotation parameters.
        const float* cov3D;
        if (!cov3D_precomp.empty())
        {
            cov3D = &cov3D_precomp[idx * 6];
        }
        else
        {
            computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds);
            cov3D = &cov3Ds[idx * 6];
        }

        // Compute 2D screen-space covariance matrix
        glm::vec3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

        // Invert covariance (EWA algorithm)
        float det = (cov.x * cov.z - cov.y * cov.y);
        if (det == 0.0f)
            continue;
        float det_inv = 1.f / det;
        glm::vec3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

        // Compute extent in screen space (by finding eigenvalues of
        // 2D covariance matrix). Use extent to compute a bounding rectangle
        // of screen-space tiles that this Gaussian overlaps with. Quit if
        // rectangle covers 0 tiles.
        float mid = 0.5f * (cov.x + cov.z);
        float lambda1 = mid + sqrt(std::max(0.1f, mid * mid - det));
        float lambda2 = mid - sqrt(std::max(0.1f, mid * mid - det));
        float my_radius = ceil(3.f * sqrt(std::max(lambda1, lambda2)));
        glm::vec2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
        glm::uvec2 rect_min, rect_max;
        getRect(point_image, my_radius, rect_min, rect_max, grid);
        if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
            continue;

        // If colors have been precomputed, use them, otherwise convert
        // spherical harmonics coefficients to RGB color.
        if (colors_precomp.empty())
        {
            glm::vec3 result = computeColorFromSH(idx, D, M, orig_points, cam_pos[0], shs, clamped);
            rgb[idx * NUM_CHANNELS + 0] = result.x;
            rgb[idx * NUM_CHANNELS + 1] = result.y;
            rgb[idx * NUM_CHANNELS + 2] = result.z;
        }

        // Store some useful helper data for the next steps.
        depths[idx] = p_view.z;
        radii[idx] = my_radius;
        points_xy_image[idx] = point_image;
        // Inverse 2D covariance and opacity neatly pack into one float4
        conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
        tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
    }
}
