#include "backward.h"
#include "auxiliary.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
    // Compute intermediate values, as it is done during forward
    glm::vec3 pos = means[idx];
    glm::vec3 dir_orig = pos - campos;
    glm::vec3 dir = dir_orig / glm::length(dir_orig);

    glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

    // Use PyTorch rule for clamping: if clamping was applied,
    // gradient becomes 0.
    glm::vec3 dL_dRGB = dL_dcolor[idx];
    dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
    dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
    dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

    glm::vec3 dRGBdx(0, 0, 0);
    glm::vec3 dRGBdy(0, 0, 0);
    glm::vec3 dRGBdz(0, 0, 0);
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    // Target location for this Gaussian to write SH gradients to
    glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

    // No tricks here, just high school-level calculus.
    float dRGBdsh0 = SH_C0;
    dL_dsh[0] = dRGBdsh0 * dL_dRGB;
    if (deg > 0)
    {
        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 = SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;
        dL_dsh[1] = dRGBdsh1 * dL_dRGB;
        dL_dsh[2] = dRGBdsh2 * dL_dRGB;
        dL_dsh[3] = dRGBdsh3 * dL_dRGB;

        dRGBdx = -SH_C1 * sh[3];
        dRGBdy = -SH_C1 * sh[1];
        dRGBdz = SH_C1 * sh[2];

        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);
            dL_dsh[4] = dRGBdsh4 * dL_dRGB;
            dL_dsh[5] = dRGBdsh5 * dL_dRGB;
            dL_dsh[6] = dRGBdsh6 * dL_dRGB;
            dL_dsh[7] = dRGBdsh7 * dL_dRGB;
            dL_dsh[8] = dRGBdsh8 * dL_dRGB;

            dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
            dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[3] * x * sh[7] + SH_C2[4] * -2.f * y * sh[8];
            dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[3] * x * sh[7] + SH_C2[4] * -2.f * z * sh[8];

            if (deg > 2)
            {
                float xxx = x * xx, xxy = x * xy, xxz = x * xz;
                float xyy = y * xy, yyz = y * yz;
                float yzz = z * yz;
                float xzz = z * xz;

                float dRGBdsh9 = SH_C3[0] * (xxx - 3.f * x * yy);
                float dRGBdsh10 = SH_C3[1] * (3.f * z * yy - yyy);
                float dRGBdsh11 = SH_C3[2] * (xxz - xyy);
                float dRGBdsh12 = SH_C3[3] * (zzz - 3.f * z * xx);
                float dRGBdsh13 = SH_C3[4] * (xxx + 3.f * y * zz - 6.f * x * xy);
                float dRGBdsh14 = SH_C3[5] * (3.f * x * yz - zzz);
                float dRGBdsh15 = SH_C3[6] * (xxz + xyy);
                float dRGBdsh16 = SH_C3[7] * (3.f * zz * y - yyy);
                float dRGBdsh17 = SH_C3[8] * (xx - yy) * z;
                dL_dsh[9] = dRGBdsh9 * dL_dRGB;
                dL_dsh[10] = dRGBdsh10 * dL_dRGB;
                dL_dsh[11] = dRGBdsh11 * dL_dRGB;
                dL_dsh[12] = dRGBdsh12 * dL_dRGB;
                dL_dsh[13] = dRGBdsh13 * dL_dRGB;
                dL_dsh[14] = dRGBdsh14 * dL_dRGB;
                dL_dsh[15] = dRGBdsh15 * dL_dRGB;
                dL_dsh[16] = dRGBdsh16 * dL_dRGB;
                dL_dsh[17] = dRGBdsh17 * dL_dRGB;

                dRGBdx += SH_C3[0] * (xx * sh[9] - 3.f * y * y * sh[9] + SH_C3[1] * (-yyy * sh[10] + 3.f * z * yy * sh[10]) + SH_C3[2] * (xxz * sh[11] - xyy * sh[11]) + SH_C3[3] * (zz * z * sh[12] - 3.f * x * x * sh[12]) + SH_C3[4] * (xxx * sh[13] + 3.f * y * zz * sh[13] - 6.f * x * xy * sh[13]) + SH_C3[5] * (3.f * x * yz * sh[14] - zzz * sh[14]) + SH_C3[6] * (xxz * sh[15] + xyy * sh[15]) + SH_C3[7] * (3.f * zz * y * sh[16] - yyy * sh[16]) + SH_C3[8] * (xx * sh[17] - yy * sh[17]) * z);
                dRGBdy += SH_C3[0] * (xy * sh[9] - x * yy * sh[9] + SH_C3[1] * (3.f * z * xy * sh[10] - xyy * sh[10]) + SH_C3[2] * (xzz * sh[11] + xyy * sh[11]) + SH_C3[3] * (xz * z * sh[12] + xx * sh[12]) + SH_C3[4] * (xxz * sh[13] + 3.f * zz * y * sh[13] - 6.f * xy * y * sh[13]) + SH_C3[5] * (3.f * xz * y * sh[14] - xzz * sh[14]) + SH_C3[6] * (xzz * sh[15] + xyy * sh[15]) + SH_C3[7] * (3.f * z * yy * sh[16] - yyy * sh[16]) + SH_C3[8] * (xx * sh[17] - yy * sh[17]) * z);
                dRGBdz += SH_C3[1] * (yy * sh[10] - 3.f * y * z * sh[10]) + SH_C3[2] * (2.f * x * xy * sh[11] + xx * sh[11] - 2.f * x * xz * sh[11]) + SH_C3[3] * (zz * sh[12] + 2.f * z * xz * sh[12] - 2.f * z * zz * sh[12]) + SH_C3[4] * (3.f * x * xx * sh[13] + 3.f * y * yy * sh[13] + 3.f * zz * zz * sh[13] - 6.f * y * xy * sh[13]) + SH_C3[5] * (xyz * sh[14] + 2.f * y * yz * sh[14] - 2.f * z * zz * sh[14]) + SH_C3[6] * (xzz * sh[15] + xyy * sh[15]) + SH_C3[7] * (3.f * z * zz * sh[16] + 2.f * y * yy * sh[16] - 2.f * y * y * z * sh[16]) + SH_C3[8] * (xx * sh[17] - yy * sh[17]) * z;
            }
        }
    }

    // Back-propagate gradients into the means.
    glm::vec3 dL_dpos = glm::vec3(
        glm::dot(glm::vec3(dRGBdx.x, dRGBdy.x, dRGBdz.x), dir),
        glm::dot(glm::vec3(dRGBdx.y, dRGBdy.y, dRGBdz.y), dir),
        glm::dot(glm::vec3(dRGBdx.z, dRGBdy.z, dRGBdz.z), dir));
    dL_dmeans[idx] = dL_dpos;
}

// Computes the energy of the loss function, and its derivatives, given
// the current set of parameters. The SH coefficients are precomputed
// at each vertex in a geometry.
double PrecomputedDiffuseModel::compute_loss(const std::vector<std::vector<double>>& SH_C1, const std::vector<std::vector<double>>& SH_C2, const std::vector<std::vector<double>>& SH_C3, const std::vector<std::vector<double>>& sh_values,
                                     const std::vector<std::vector<double>>& sh_d1, const std::vector<std::vector<double>>& sh_d2, const std::vector<std::vector<double>>& sh_d3,
                                     const std::vector<glm::vec3>& points, const std::vector<double>& sh_norm,
                                     const std::vector<double>& intensity, const std::vector<double>& mean,
                                     double lambda,
                                     std::vector<glm::vec3>& dL_dmeans)
{
    double L = 0.f;
    for (size_t i = 0; i < points.size(); ++i)
    {
        glm::vec3 pos = points[i] - glm::vec3(mean[0], mean[1], mean[2]);
        glm::normalize(pos);
        std::vector<double> sh = sh_values[i];
        std::vector<double> sh_d = sh_d1[i];
        glm::vec3 N(sh_d[0], sh_d[1], sh_d[2]);
        glm::normalize(N);
        glm::vec3 dL_dRGB(0.0f, 0.0f, 0.0f);
        std::vector<double> sh2 = sh_d2[i];
        std::vector<double> sh3 = sh_d3[i];
        compute_grads(N, sh, sh2, sh3, SH_C1, SH_C2, SH_C3, dL_dRGB);
        L += energy_loss(sh, intensity[i], lambda);
        dL_dmeans[i] = glm::vec3(0.0f, 0.0f, 0.0f);

        // Call the compute_gradients function to fill in dL_dmeans.
        compute_gradients(pos, sh, sh2, sh3, SH_C1, SH_C2, SH_C3, dL_dRGB, dL_dmeans, i);
    }
    return L;
}
