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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <functional>
#include <glm/glm.hpp>

#define BLOCK_X 16
#define BLOCK_Y 16

namespace CpuRasterizer
{
    struct GeometryState
    {
        size_t scan_size;
        float* depths;
        bool* clamped;
        int* internal_radii;
        glm::vec2* means2D;
        float* cov3D;
        glm::vec4* conic_opacity;
        float* rgb;
        uint32_t* point_offsets;
        uint32_t* tiles_touched;

        static GeometryState fromChunk(char*& chunk, size_t P);
    };

    struct ImageState
    {
        uint2* ranges;
        uint32_t* n_contrib;
        float* accum_alpha;

        static ImageState fromChunk(char*& chunk, size_t N);
    };

    struct BinningState
    {
        size_t sorting_size;
        uint64_t* point_list_keys_unsorted;
        uint64_t* point_list_keys;
        uint32_t* point_list_unsorted;
        uint32_t* point_list;
        char* list_sorting_space;

        static BinningState fromChunk(char*& chunk, size_t P);
    };

    template<typename T> 
    size_t required(size_t P)
    {
        char* size = nullptr;
        T::fromChunk(size, P);
        return ((size_t)size) + 128;
    }

    uint32_t getHigherMsb(uint32_t n)
    {
        uint32_t msb = sizeof(n) * 4;
        uint32_t step = msb;
        while (step > 1)
        {
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

    void checkFrustum(
        int P,
        const float* orig_points,
        const float* viewmatrix,
        const float* projmatrix,
        bool* present)
    {
        for (int idx = 0; idx < P; ++idx)
        {
            float3 p_view;
            present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
        }
    }

    void duplicateWithKeys(
        int P,
        const glm::vec2* points_xy,
        const float* depths,
        const uint32_t* offsets,
        uint64_t* gaussian_keys_unsorted,
        uint32_t* gaussian_values_unsorted,
        int* radii,
        const glm::ivec2& grid)
    {
        for (int idx = 0; idx < P; ++idx)
        {
            if (radii[idx] > 0)
            {
                uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
                glm::ivec2 rect_min, rect_max;

                getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

                for (int y = rect_min.y; y < rect_max.y; y++)
                {
                    for (int x = rect_min.x; x < rect_max.x; x++)
                    {
                        uint64_t key = y * grid.x + x;
                        key <<= 32;
                        key |= *((uint32_t*)&depths[idx]);
                        gaussian_keys_unsorted[off] = key;
                        gaussian_values_unsorted[off] = idx;
                        off++;
                    }
                }
            }
        }
    }

    void identifyTileRanges(
        int L,
        uint64_t* point_list_keys,
        uint2* ranges)
    {
        for (int idx = 0; idx < L; ++idx)
        {
            uint64_t key = point_list_keys[idx];
            uint32_t currtile = key >> 32;
            if (idx == 0)
                ranges[currtile].x = 0;
            else
            {
                uint32_t prevtile = point_list_keys[idx - 1] >> 32;
                if (currtile != prevtile)
                {
                    ranges[prevtile].y = idx;
                    ranges[currtile].x = idx;
                }
            }
            if (idx == L - 1)
                ranges[currtile].y = L;
        }
    }

    GeometryState GeometryState::fromChunk(char*& chunk, size_t P)
    {
        GeometryState geom;
        geom.depths = reinterpret_cast<float*>(chunk);
        chunk += P * sizeof(float);
        geom.clamped = reinterpret_cast<bool*>(chunk);
        chunk += P * 3 * sizeof(bool);
        geom.internal_radii = reinterpret_cast<int*>(chunk);
        chunk += P * sizeof(int);
        geom.means2D = reinterpret_cast<glm::vec2*>(chunk);
        chunk += P * sizeof(glm::vec2);
        geom.cov3D = reinterpret_cast<float*>(chunk);
        chunk += P * 6 * sizeof(float);
        geom.conic_opacity = reinterpret_cast<glm::vec4*>(chunk);
        chunk += P * sizeof(glm::vec4);
        geom.rgb = reinterpret_cast<float*>(chunk);
        chunk += P * 3 * sizeof(float);
        geom.tiles_touched = reinterpret_cast<uint32_t*>(chunk);
        chunk += P * sizeof(uint32_t);
        geom.scan_size = chunk - reinterpret_cast<char*>(geom.tiles_touched);
        geom.scanning_space = reinterpret_cast<char*>(geom.tiles_touched + P);
        geom.point_offsets = reinterpret_cast<uint32_t*>(geom.scanning_space);
        return geom;
    }

    ImageState ImageState::fromChunk(char*& chunk, size_t N)
    {
        ImageState img;
        img.accum_alpha = reinterpret_cast<float*>(chunk);
        chunk += N * sizeof(float);
        img.n_contrib = reinterpret_cast<uint32_t*>(chunk);
        chunk += N * sizeof(uint32_t);
        img.ranges = reinterpret_cast<uint2*>(chunk);
        return img;
    }

    BinningState BinningState::fromChunk(char*& chunk, size_t P)
    {
        BinningState binning;
        binning.point_list = reinterpret_cast<uint32_t*>(chunk);
        chunk += P * sizeof(uint32_t);
        binning.point_list_unsorted = reinterpret_cast<uint32_t*>(chunk);
        chunk += P * sizeof(uint32_t);
        binning.point_list_keys = reinterpret_cast<uint64_t*>(chunk);
        chunk += P * sizeof(uint64_t);
        binning.point_list_keys_unsorted = reinterpret_cast<uint64_t*>(chunk);
        chunk += P * sizeof(uint64_t);
        binning.sorting_size = chunk - reinterpret_cast<char*>(binning.point_list_keys_unsorted);
        binning.list_sorting_space = reinterpret_cast<char*>(binning.point_list_keys_unsorted + P);
        return binning;
    }

    void Rasterizer::markVisible(
        int P,
        float* means3D,
        float* viewmatrix,
        float* projmatrix,
        bool* present)
    {
        checkFrustum(P, means3D, viewmatrix, projmatrix, present);
    }

    int Rasterizer::forward(
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
        bool debug)
    {
        const float focal_y = height / (2.0f * tan_fovy);
        const float focal_x = width / (2.0f * tan_fovx);

        size_t chunk_size = required<GeometryState>(P);
        char* chunkptr = geometryBuffer(chunk_size);
        GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

        if (radii == nullptr)
        {
            radii = geomState.internal_radii;
        }

        glm::ivec2 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);

        size_t img_chunk_size = required<ImageState>(width * height);
        char* img_chunkptr = imageBuffer(img_chunk_size);
        ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

        std::vector<uint32_t> tiles_touched(P);
        std::partial_sum(geomState.tiles_touched, geomState.tiles_touched + P, tiles_touched.begin());

        int num_rendered = tiles_touched.back();
        size_t binning_chunk_size = required<BinningState>(num_rendered);
        char* binning_chunkptr = binningBuffer(binning_chunk_size);
        BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

        duplicateWithKeys(P, geomState.means2D, geomState.depths, geomState.point_offsets,
            binningState.point_list_keys_unsorted, binningState.point_list_unsorted, radii, tile_grid);

        int bit = getHigherMsb(tile_grid.x * tile_grid.y);

        std::sort(binningState.point_list_keys_unsorted, binningState.point_list_keys_unsorted + num_rendered);
        std::vector<uint2> ranges(tile_grid.x * tile_grid.y);

        identifyTileRanges(num_rendered, binningState.point_list_keys, ranges.data());

        for (int idx = 0; idx < P; ++idx)
        {
            geomState.point_offsets[idx] = tiles_touched[idx];
        }

        const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;

        render(tile_grid, imgState.ranges, binningState.point_list, width, height, geomState.means2D,
            feature_ptr, geomState.conic_opacity, imgState.accum_alpha, imgState.n_contrib, background, out_color);

        return num_rendered;
    }

    void Rasterizer::backward(
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
        bool debug)
    {
        GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
        BinningState binningState = BinningState::fromChunk(binning_buffer, R);
        ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

        if (radii == nullptr)
        {
            radii = geomState.internal_radii;
        }

        const float focal_y = height / (2.0f * tan_fovy);
        const float focal_x = width / (2.0f * tan_fovx);

        glm::ivec2 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);

        render(tile_grid, imgState.ranges, binningState.point_list, width, height, geomState.means2D,
            colors_precomp, geomState.conic_opacity, imgState.accum_alpha, imgState.n_contrib, background, dL_dpix);

        preprocess(P, D, M, means3D, radii, shs, geomState.clamped, scales, rotations, scale_modifier, cov3D_precomp,
            viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy, campos,
            dL_dmean2D, dL_dconic, dL_dmean3D, dL_dcolor, dL_dcov3D, dL_dsh, dL_dscale, dL_drot);
    }
};
