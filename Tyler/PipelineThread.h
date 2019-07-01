#pragma once

#include "RasterizerConfig.h"
#include "RenderState.h"

namespace tyler
{
    struct RenderEngine;
    struct CoverageMask;

    // Thread execution state
    enum class ThreadStatus : uint8_t
    {
        IDLE,                               // Waiting for input arrival by RenderEngine
        DRAWCALL_TOP,                       // Input data received, start processing drawcall
        DRAWCALL_GEOMETRY,                  // Geometry processing in progress
        DRAWCALL_BINNING,                   // Binning in progress
        DRAWCALL_SYNC_POINT_POST_BINNER,    // Sync post binning
        DRAWCALL_RASTERIZATION,             // Rasterization in progress
        DRAWCALL_SYNC_POINT_POST_RASTER,    // Sync post rasterization
        DRAWCALL_FRAGMENTSHADER,            // Fragment processing in progress
        DRAWCALL_BOTTOM,                    // Drawcall processed
        TERMINATED                          // Thread shut down requested
    };

    struct PipelineThread
    {
        PipelineThread(RenderEngine* pRenderEngine, uint32_t threadIdx);
        ~PipelineThread();

        // Worker thread procedure
        void Run();

        // Process received drawcall input
        void ProcessDrawcall();

        // Vertex Shader
        void ExecuteVertexShader(uint32_t drawIdx, uint32_t primIdx, glm::vec4* pV0Clip, glm::vec4* pV1Clip, glm::vec4* pV2Clip);

        // Clipper (full-triangle only)
        bool ExecuteFullTriangleClipping(uint32_t primIdx, const glm::vec4& v0Clip, const glm::vec4& v1Clip, const glm::vec4& v2Clip);

        // Triangle Setup + Culling
        bool ExecuteTriangleSetupAndCull(uint32_t primIdx, const glm::vec4& v0Clip, const glm::vec4& v1Clip, const glm::vec4& v2Clip);

        // Binner
        void ExecuteBinner(uint32_t primIdx, const glm::vec4& v0Clip, const glm::vec4& v1Clip, const glm::vec4& v2Clip);

        // Rasterizer
        void ExecuteRasterizer();

        // Fragment Shading
        void ExecuteFragmentShader();

        void FragmentShadeTile(uint32_t tilePosX, uint32_t tilePosY, uint32_t primIdx);
        void FragmentShadeBlock(uint32_t blockPosX, uint32_t blockPosY, uint32_t primIdx);
        void FragmentShadeQuad(CoverageMask* pMask);        

        // Given three clip-space verices, compute the bounding box of a triangle clamped to width/height
        void ComputeBoundingBox(const glm::vec4& v0Clip, const glm::vec4& v1Clip, const glm::vec4& v2Clip, float width, float height, Rect2D* pBbox) const;

        // Calculate interpolation coefficients to be used during FS
        // to calculate perspective-correct interpolation of vertex attributes
        void CalculateInterpolationCoefficients(
            uint32_t drawIDx,
            const VertexAttributes& vertexAttribs0,
            const VertexAttributes& vertexAttribs1,
            const VertexAttributes& vertexAttribs2);

        // Compute interpolation basis functions f0(x,y) & f1(x,y)
        void ComputeParameterBasisFunctions(
            uint32_t sampleX,
            uint32_t sampleY,
            const __m128& sseA4Edge0,
            const __m128& sseB4Edge0,
            const __m128& sseC4Edge0,
            const __m128& sseA4Edge1,
            const __m128& sseB4Edge1,
            const __m128& sseC4Edge1,
            const __m128& sseA4Edge2,
            const __m128& sseB4Edge2,
            const __m128& sseC4Edge2,
            __m128* pSSEf0XY,
            __m128* pSSEf1XY);

        // Using basis functions, interpolated Z values (for depth test)
        void InterpolateDepthValues(
            uint32_t primIdx,
            const __m128& ssef0XY,
            const __m128& ssef1XY,
            __m128* pZInterpolated);

        // Using basis functions computed already, interpolate each attribute channel present
        void InterpolateVertexAttributes(
            uint32_t primIdx,
            const __m128& ssef0XY,
            const __m128& ssef1XY,
            InterpolatedAttributes* pInterpolationAttributes);

        // Utilities for VS$
        bool PerformVertexCacheLookup(uint32_t primIdx, uint32_t* pCachedIdx);
        void CacheVertexData(uint32_t vertexIdx, const glm::vec4& vClip, const tyler::VertexAttributes& tempVertexAttrib);
        void CopyVertexData(glm::vec4* pVClip, uint32_t cacheEntry, VertexAttributes* pTempVertexAttrib);

        // Unique RenderEngine instance
        RenderEngine*               m_pRenderEngine = nullptr;
        const RasterizerConfig&     m_RenderConfig;

        // Unique thread index among all PipelineThreads created
        uint32_t                    m_ThreadIdx;

        // Underlying thread that will be used to execute all stages of the pipeline in order
        std::thread                 m_WorkerThread;

        // Thread execution state
        std::atomic<ThreadStatus>   m_CurrentState;

        // VS$ entry
        struct VertexCache
        {
            // Clip-space position of vertex
            glm::vec4           m_ClipPos;

            // Vertex attributes of vertex
            VertexAttributes    m_VertexAttribs;
        }                       m_VertexCacheEntries[g_scVertexShaderCacheSize];

        // Intermediate vertex attributes used for VS invocations
        VertexAttributes            m_TempVertexAttributes[3];

        // Array of indices of cached vertices
        uint32_t                    m_CachedVertexIndices[g_scVertexShaderCacheSize];
        // Number of vertices currently cached
        uint32_t                    m_NumVertexCacheEntries;

        // Per-drawcall data, to be prepared by RenderEngine
        // before a drawcall arrival will be issued to a thread
        struct DrawParams
        {
            // Start and end indices into the drawcall data (vertex/index buffer)
            // To be sliced (per-thread) and sized (per-iteration) appropriately 
            uint32_t                m_ElemsStart = 0u;
            uint32_t                m_ElemsEnd = 0u;
            // Vertex offset to be added to base vertex index
            uint32_t                m_VertexOffset = 0u;
        }                           m_ActiveDrawParams;
    };
}