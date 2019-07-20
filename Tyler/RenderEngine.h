#pragma once

#include "RasterizerConfig.h"
#include "RenderState.h"
#include "TileQueue.h"
#include "CoverageMaskBuffer.h"

namespace tyler
{
    struct PipelineThread;

    struct TriangleSetupBuffers
    {
        // Coefficients of three edge equations
        glm::vec3*  m_pEdgeCoefficients;

        // Interpolated z coordinates of three vertices
        glm::vec3*  m_pInterpolatedZValues;

        // Interpolation deltas computed after VS that'll be used for perspective-correct interpolation of vertex attributes
        glm::vec3*  m_Attribute4Deltas[g_scMaxVertexAttributes];
        glm::vec3*  m_Attribute3Deltas[g_scMaxVertexAttributes];
        glm::vec3*  m_Attribute2Deltas[g_scMaxVertexAttributes];

        // Cache bounding boxes computed during binning for use in rasterization
        Rect2D*     m_pPrimBBoxes;
    };

    struct RenderEngine
    {
        RenderEngine(const RasterizerConfig& renderConfig);
        ~RenderEngine();

        // Fully clear bound color and depth buffers before starting a render pass
        void ClearRenderTargets(bool clearColor, const glm::vec4& colorValue, bool clearDepth, float depthValue);

        // Bind active framebuffer and allocate RT-dependent data, if necessary (e.g. RT resolution change, NULL RT, etc.)
        void SetRenderTargets(Framebuffer* pFramebuffer);

        // Draw the object by using bound pipeline states
        void DrawIndexed(uint32_t indexCount, uint32_t vertexOffset);

        // Perform necessary state/data invalidations for drawcalls and draw iterations
        void ApplyPreDrawcallStateInvalidations();
        void ApplyPreDrawIterationStateInvalidations();

        // Stall callee until all PipelineThreads complete processing of a single drawcall
        void WaitForPipelineThreadsToCompleteProcessingDrawcall() const;

        // Stall PipelineThreads until all of them complete binning primitives to their respective tiles
        void WaitForPipelineThreadsToCompleteBinning() const;

        // Stall PipelineThreads until all of them complete rasterization
        void WaitForPipelineThreadsToCompleteRasterization() const;

        // Return a tile's global index given its row/column address
        uint32_t GetGlobalTileIndex(uint32_t tileX, uint32_t tileY) const
        {
            return (tileX + tileY * m_NumTilePerRow);
        }

        // Return next available tile index to be rasterized from tile queue
        uint32_t FetchNextTileForRasterization()
        {
            return m_RasterizerQueue.FetchNextTileIndex();
        }

        // Return next available tile index to be fragment-shaded from tile queue
        uint32_t FetchNextTileForFragmentShading()
        {
            return m_RasterizerQueue.RemoveTileIndex();
        }

        // Add the tile to the rasterizer queue iff it's not done yet
        void EnqueueTileForRasterization(uint32_t tileIdx);

        // Bin a primitive to thread-local bin of a tile
        void BinPrimitiveForTile(uint32_t threadIdx, uint32_t tileIdx, uint32_t primIdx);

        // Append tile, block or fragment coverage mask
        void AppendCoverageMask(uint32_t threadIdx, uint32_t tileIdx, const CoverageMask& mask);

        // Check and grow mask buffers if needed
        void ResizeCoverageMaskBuffer(uint32_t threadIdx, uint32_t tileIdx);

        // Write interpolated Z values to depth buffer based on write mask at given sample
        void UpdateDepthBuffer(const __m128& sseWriteMask, const __m128& sseDepthValues, uint32_t sampleX, uint32_t sampleY);

        // Fetch depth buffer contents at given sample
        __m128 FetchDepthBuffer(uint32_t sampleX, uint32_t sampleY) const;

        // Write shaded fragment output to color buffer based on write mask at given sample
        void UpdateColorBuffer(const __m128& sseWriteMask, const FragmentOutput& fragmentOutput, uint32_t sampleX, uint32_t sampleY);

        // Global rendering parameters
        const RasterizerConfig&                         m_RenderConfig;

        // Active frame buffer configuration
        Framebuffer                                     m_Framebuffer;

        // Bound vertex buffer
        VertexBuffer*                                   m_pVertexBuffer = nullptr;
        // Vertex input stride in bytes
        uint32_t                                        m_VertexInputStride = 0u;

        // Bound index buffer
        IndexBuffer*                                    m_pIndexBuffer = nullptr;

        // SoA for all data required for TriangleSetup
        TriangleSetupBuffers                            m_SetupBuffers;

        // Bound Vertex & Fragment shader function pointers that will be invoked
        VertexShader                                    m_VertexShader = nullptr;
        FragmentShader                                  m_FragmentShader = nullptr;
        ConstantBuffer*                                 m_pConstantBuffer = nullptr;
        ShaderMetadata                                  m_ShaderMetadata;

        // PipelineThreads will run concurrently to implement the pipeline stages
        std::vector<PipelineThread*>                    m_PipelineThreads;

        // Array of tiles that'll be allocated based on screen resolution and fixed tile size
        std::vector<Tile>                               m_TileList;

        // FIFO of tiles waiting to be rasterized
        TileQueue                                       m_RasterizerQueue;

        // Array of per-tile bins which contain indices of primitives that intersect a tile
        std::vector<std::vector<std::vector<uint32_t>>> m_BinList;

        // Per-thread array of tile coverage masks emitted by rasterizers concurrently
        std::vector<std::vector<CoverageMaskBuffer*>>   m_CoverageMasks;

        // Number of tiles per row/column
        uint32_t                                        m_NumTilePerRow = 0u;
        uint32_t                                        m_NumTilePerColumn = 0u;
    };
}