#pragma once

namespace tyler
{
    // Global list of compile-time pipeline configurations

    // Toggle full-triangle clipping before binning
    static constexpr bool       g_scFullTriangleClippingEnabled = true;

    // Toggle VS$
    static constexpr bool       g_scVertexShaderCacheEnabled = true;

    // VS$ max entry size per-thread
    static constexpr uint32_t   g_scVertexShaderCacheSize = 32u;

    // All tiles consist of blocks which are groups of 8x8 pixels
    static constexpr uint32_t   g_scPixelBlockSize = 8u;

    // SSE -> 4 | AVX -> 8
    static constexpr uint32_t   g_scSIMDWidth = 4u;

    // # samples per row / SIMD width
    static constexpr uint32_t   g_scNumEdgeTestsPerRow = g_scPixelBlockSize / g_scSIMDWidth;

    // Initial coverage masks buffer size
    static constexpr uint32_t   g_scRasterizerCoverageMaskBufferInitialSize = 4096u;

    // Tiles comprise of one or more blocks which are a fixed-size 8x8 group of pixels
    enum TileSize : uint32_t
    {
        TILE_SIZE_MIN = 8u,
        TILE_SIZE_8x8 = 8u,         // 1x1 block
        TILE_SIZE_16x16 = 16u,      // 2x2 blocks
        TILE_SIZE_32x32 = 32u,      // 4x4 blocks
        TILE_SIZE_64x64 = 64u,      // 8x8 blocks
        TILE_SIZE_128x128 = 128u,   // 16x16 blocks
        TILE_SIZE_256x256 = 256u,   // 32x32 blocks
        TILE_SIZE_512x512 = 512u,   // 64x64 blocks
        TILE_SIZE_MAX = 512
    };

    struct RasterizerConfig
    {
        // List of all runtime/algorithmic parameters that can be configurad via command line

        // Number of pipeline threads that will be created to implement pipeline stages in parallel.
        // @default: Max # of concurrent HW threads supported on given machine
        uint32_t    m_NumPipelineThreads = 0u;

        // Max number of primitives to be processed per-iteration
        // @default: 6000 prims
        uint32_t    m_MaxDrawIterationSize = 6000u;

        // Frame buffer tile size, multiples of 8x8 block(s) of pixels
        // @default: 64x64 == 8x8 blocks
        uint32_t    m_TileSize = TILE_SIZE_64x64;
    };
}