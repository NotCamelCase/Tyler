#pragma once

#include "RasterizerConfig.h"
#include "RenderState.h"

namespace tyler
{
    struct RenderEngine;
    struct CoverageMask;

    // POD struct to pass SIMD registers initialized with EE coefficients to fragment-shader routines more easily
    struct SIMDEdgeCoefficients
    {
        __m128  m_SSEA4Edge0;
        __m128  m_SSEA4Edge1;
        __m128  m_SSEA4Edge2;

        __m128  m_SSEB4Edge0;
        __m128  m_SSEB4Edge1;
        __m128  m_SSEB4Edge2;

        __m128  m_SSEC4Edge0;
        __m128  m_SSEC4Edge1;
        __m128  m_SSEC4Edge2;
    };

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
        template<bool IsIndexed>
        void ProcessDrawcall();

        // Vertex Shader
        template<bool IsIndexed>
        void ExecuteVertexShader(uint32_t drawIdx, uint32_t primIdx, glm::vec4* pV0Clip, glm::vec4* pV1Clip, glm::vec4* pV2Clip);

        // Clipper (full-triangle only)
        bool ExecuteFullTriangleClipping(uint32_t primIdx, const glm::vec4& v0Clip, const glm::vec4& v1Clip, const glm::vec4& v2Clip, Rect2D* pBbox);

        // Triangle Setup + Culling
        bool ExecuteTriangleSetupAndCull(uint32_t primIdx, const glm::vec4& v0Clip, const glm::vec4& v1Clip, const glm::vec4& v2Clip);

        // Binner
        void ExecuteBinner(uint32_t primIdx, const glm::vec4& v0Clip, const glm::vec4& v1Clip, const glm::vec4& v2Clip, const Rect2D& bbox);

        // Rasterizer
        void ExecuteRasterizer();

        // Fragment Shading
        void ExecuteFragmentShader();

        // Fragment shader routines at tile/block/fragment levels
        void FragmentShadeTile(
            uint32_t tilePosX,
            uint32_t tilePosY,
            uint32_t primIdx,
            const SIMDEdgeCoefficients& simdEERegs);

        void FragmentShadeBlock(
            uint32_t blockPosX,
            uint32_t blockPosY,
            uint32_t primIdx,
            const SIMDEdgeCoefficients& simdEERegs);

        void FragmentShadeQuad(
            CoverageMask* pMask,
            const SIMDEdgeCoefficients& simdEERegs);

        // Given three clip-space verices, compute the bounding box of a triangle clamped to width/height
        Rect2D ComputeBoundingBox(const glm::vec4& v0Clip, const glm::vec4& v1Clip, const glm::vec4& v2Clip, float width, float height) const;

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
            const SIMDEdgeCoefficients& simdEERegs,
            __m128* pSSEf0XY,
            __m128* pSSEf1XY);

        // Using basis functions, interpolated Z values (for depth test)
        __m128 InterpolateDepthValues(
            uint32_t primIdx,
            const __m128& ssef0XY,
            const __m128& ssef1XY);

        // Using basis functions computed already, interpolate each attribute channel present
        void InterpolateVertexAttributes(
            uint32_t primIdx,
            const __m128& ssef0XY,
            const __m128& ssef1XY,
            InterpolatedAttributes* pInterpolationAttributes);

        // Utilities for VS$
        bool PerformVertexCacheLookup(uint32_t primIdx, uint32_t* pCachedIdx);
        void CacheVertexData(uint32_t vertexIdx, const glm::vec4& vClip, const tyler::VertexAttributes& tempVertexAttrib);
        void CopyVertexData(uint32_t cacheEntry, glm::vec4* pVClip, VertexAttributes* pTempVertexAttrib);

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
            // Drawcall is indexed or not
            bool                    m_IsIndexed = false;
        }                           m_ActiveDrawParams;
    };
}