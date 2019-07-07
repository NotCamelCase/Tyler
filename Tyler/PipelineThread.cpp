#include "PipelineThread.h"

#include "RenderEngine.h"
#include "RenderState.h"

namespace tyler
{
    PipelineThread::PipelineThread(RenderEngine* pRenderEngine, uint32_t threadIdx)
        :
        m_pRenderEngine(pRenderEngine),
        m_RenderConfig(m_pRenderEngine->m_RenderConfig),
        m_ThreadIdx(threadIdx),
        m_CurrentState(ThreadStatus::IDLE)
    {
        m_WorkerThread = std::thread(&PipelineThread::Run, this);
    }

    PipelineThread::~PipelineThread()
    {
        m_CurrentState.store(ThreadStatus::TERMINATED);

        ASSERT(m_WorkerThread.joinable());
        m_WorkerThread.join();
    }

    void PipelineThread::Run()
    {
        while (m_CurrentState.load(std::memory_order_relaxed) != ThreadStatus::TERMINATED)
        {
            if (m_CurrentState.load(std::memory_order_relaxed) == ThreadStatus::DRAWCALL_TOP)
            {
                // Drawcall received, switch to processing it
                ProcessDrawcall();
            }
        }
    }

    void PipelineThread::ProcessDrawcall()
    {
        LOG("Thread %d drawcall processing begins\n", m_ThreadIdx);

        // Drawcall starts with geometry processing
        m_CurrentState.store(ThreadStatus::DRAWCALL_GEOMETRY, std::memory_order_relaxed);

        LOG("Thread %d processing geometry...\n", m_ThreadIdx);

        // Iterate over triangles in assigned drawcall range
        for (uint32_t drawIdx = m_ActiveDrawParams.m_ElemsStart, primIdx = m_ActiveDrawParams.m_ElemsStart % m_RenderConfig.m_MaxDrawIterationSize;
            drawIdx < m_ActiveDrawParams.m_ElemsEnd;
            drawIdx++, primIdx++)
        {
            // drawIdx = Assigned prim indices which will be only used to fetch indices
            // primIdx = Prim index relative to current iteration

            // Clip-space vertices to be retrieved from VS
            glm::vec4 v0Clip, v1Clip, v2Clip;

            // VS
            ExecuteVertexShader(drawIdx, primIdx, &v0Clip, &v1Clip, &v2Clip);

            // CLIPPER
            if (!ExecuteFullTriangleClipping(primIdx, v0Clip, v1Clip, v2Clip))
            {
                // Triangle clipped, proceed iteration with next primitive
                continue;
            }

            // TRIANGLE SETUP & CULL
            if (!ExecuteTriangleSetupAndCull(primIdx, v0Clip, v1Clip, v2Clip))
            {
                // Triangle culled, proceed iteration with next primitive
                continue;
            }

            // BINNER
            ExecuteBinner(primIdx, v0Clip, v1Clip, v2Clip);
        }

        ASSERT(m_CurrentState.load() <= ThreadStatus::DRAWCALL_BINNING);

        LOG("Thread %d post-binning sync point\n", m_ThreadIdx);

        // To preserve rendering order, we must ensure that all threads finish binning primitives to tiles
        // before rasterization is started. To do that, we will stall all threads to sync @DRAWCALL_RASTERIZATION

        // All stages up to binning completed, set state to post binning
        // and stall until all PipelineThreads complete binning
        m_CurrentState.store(ThreadStatus::DRAWCALL_SYNC_POINT_POST_BINNER, std::memory_order_release);
        m_pRenderEngine->WaitForPipelineThreadsToCompleteBinning();

        LOG("Thread %d post-binning sync point reached!\n", m_ThreadIdx);

        // State must have been set to rasterization by RenderEngine
        // when binnnig is "signaled" to have ended
        ASSERT(m_CurrentState.load() == ThreadStatus::DRAWCALL_RASTERIZATION);

        LOG("Thread %d rasterizing...\n", m_ThreadIdx);

        // RASTERIZATION
        ExecuteRasterizer();

        // Rasterization completed, set state to post raster and
        // stall until all PipelineThreads complete rasterization.
        // We need this sync because when (N-x) threads finish rasterization and
        // reach the end of tile queue while x threads are still busy rasterizing tile blocks,
        // we must ensure that none of the (N-x) non-busy threads will go ahead and start fragment-shading tiles
        // whose blocks could be currently still rasterized by x remaining threads

        LOG("Thread %d post-raster sync point\n", m_ThreadIdx);

        // All stages up to rasterization completed, set state to post raster
        // and stall until all PipelineThreads complete rasterization
        m_CurrentState.store(ThreadStatus::DRAWCALL_SYNC_POINT_POST_RASTER, std::memory_order_release);
        m_pRenderEngine->WaitForPipelineThreadsToCompleteRasterization();

        LOG("Thread %d post-raster sync point reached!\n", m_ThreadIdx);

        // State must have been set to fragment shader by RenderEngine
        // when rasterization is "signaled" to have ended
        ASSERT(m_CurrentState.load() == ThreadStatus::DRAWCALL_FRAGMENTSHADER);

        LOG("Thread %d fragment-shading...\n", m_ThreadIdx);

        // FS
        ExecuteFragmentShader();

        LOG("Thread %d drawcall ended\n", m_ThreadIdx);

        // Draw iteration completed
        m_CurrentState.store(ThreadStatus::DRAWCALL_BOTTOM, std::memory_order_relaxed);
    }

    void PipelineThread::ExecuteVertexShader(uint32_t drawIdx, uint32_t primIdx, glm::vec4* pV0Clip, glm::vec4* pV1Clip, glm::vec4* pV2Clip)
    {
        uint8_t* pVertexBuffer = static_cast<uint8_t*>(m_pRenderEngine->m_pVertexBuffer);
        IndexBuffer* pIndexBuffer = m_pRenderEngine->m_pIndexBuffer;
        ASSERT((pVertexBuffer != nullptr) && (pIndexBuffer != nullptr)); //TODO: Non-indexed DC!

        ConstantBuffer* pConstantBuffer = m_pRenderEngine->m_pConstantBuffer;

        uint32_t vertexStride = m_pRenderEngine->m_VertexInputStride;
        uint32_t vertexOffset = m_ActiveDrawParams.m_VertexOffset;

        VertexAttributes* pTempVertexAttrib0 = &m_TempVertexAttributes[0];
        VertexAttributes* pTempVertexAttrib1 = &m_TempVertexAttributes[1];
        VertexAttributes* pTempVertexAttrib2 = &m_TempVertexAttributes[2];

        VertexShader VS = m_pRenderEngine->m_VertexShader;
        ASSERT(VS != nullptr);

        if constexpr (!g_scVertexShaderCacheEnabled)
        {
            // VS$ disabled, don't look up vertices in the cache

            // Fetch pointers to vertex input that'll be passed to vertex shader
            uint8_t* pVertIn0 = &pVertexBuffer[vertexStride * pIndexBuffer[vertexOffset + (3 * drawIdx + 0)]];
            uint8_t* pVertIn1 = &pVertexBuffer[vertexStride * pIndexBuffer[vertexOffset + (3 * drawIdx + 1)]];
            uint8_t* pVertIn2 = &pVertexBuffer[vertexStride * pIndexBuffer[vertexOffset + (3 * drawIdx + 2)]];

            // Invoke vertex shader with vertex attributes payload
            *pV0Clip = VS(pVertIn0, pTempVertexAttrib0, pConstantBuffer);
            *pV1Clip = VS(pVertIn1, pTempVertexAttrib1, pConstantBuffer);
            *pV2Clip = VS(pVertIn2, pTempVertexAttrib2, pConstantBuffer);
        }
        else
        {
            uint32_t cacheEntry0 = UINT32_MAX;
            uint32_t cacheEntry1 = UINT32_MAX;
            uint32_t cacheEntry2 = UINT32_MAX;

            uint32_t vertexIdx0 = pIndexBuffer[vertexOffset + (3 * drawIdx + 0)];
            uint32_t vertexIdx1 = pIndexBuffer[vertexOffset + (3 * drawIdx + 1)];
            uint32_t vertexIdx2 = pIndexBuffer[vertexOffset + (3 * drawIdx + 2)];

            if (PerformVertexCacheLookup(vertexIdx0, &cacheEntry0))
            {
                // Vertex 0 is found in the cache, skip VS and fetch cached data
                CopyVertexData(pV0Clip, cacheEntry0, pTempVertexAttrib0);
            }
            else
            {
                // Vertex 0 is not found in the cache,
                // first invoke VS and then cache the clip-space position & vertex attributes

                uint8_t* pVertIn0 = &pVertexBuffer[vertexStride * vertexIdx0];
                *pV0Clip = VS(pVertIn0, pTempVertexAttrib0, pConstantBuffer);

                CacheVertexData(vertexIdx0, *pV0Clip, *pTempVertexAttrib0);
            }

            if (PerformVertexCacheLookup(vertexIdx1, &cacheEntry1))
            {
                // Vertex 1 is found in the cache, skip VS and fetch cached data
                CopyVertexData(pV1Clip, cacheEntry1, pTempVertexAttrib1);
            }
            else
            {
                // Vertex 1 is not found in the cache,
                // first invoke VS and then cache the clip-space position & vertex attributes

                uint8_t* pVertIn1 = &pVertexBuffer[vertexStride * vertexIdx1];
                *pV1Clip = VS(pVertIn1, pTempVertexAttrib1, pConstantBuffer);

                CacheVertexData(vertexIdx1, *pV1Clip, *pTempVertexAttrib1);
            }

            
            if (PerformVertexCacheLookup(vertexIdx2, &cacheEntry2))
            {
                // Vertex 2 is found in the cache, skip VS and fetch cached data
                CopyVertexData(pV2Clip, cacheEntry2, pTempVertexAttrib2);
            }
            else
            {
                // Vertex 2 is not found in the cache,
                // first invoke VS and then cache the clip-space position & vertex attributes

                uint8_t* pVertIn2 = &pVertexBuffer[vertexStride * vertexIdx2];
                *pV2Clip = VS(pVertIn2, pTempVertexAttrib2, pConstantBuffer);

                CacheVertexData(vertexIdx0, *pV2Clip, *pTempVertexAttrib2);
            }
        }

        // Calculate interpolation data for active vertex attributes
        CalculateInterpolationCoefficients(primIdx, *pTempVertexAttrib0, *pTempVertexAttrib1, *pTempVertexAttrib2);
    }

    void PipelineThread::CopyVertexData(glm::vec4* pVClip, uint32_t cacheEntry, VertexAttributes* pTempVertexAttrib)
    {
        // Copy cached clip-space positions
        *pVClip = m_VertexCacheEntries[cacheEntry].m_ClipPos;

        // Copy vertex (only active!) attributes
        memcpy(
            pTempVertexAttrib->m_Attributes2,
            m_VertexCacheEntries[cacheEntry].m_VertexAttribs.m_Attributes2,
            sizeof(glm::vec2) * m_pRenderEngine->m_ShaderMetadata.m_NumVec2Attributes);

        memcpy(
            pTempVertexAttrib->m_Attributes3,
            m_VertexCacheEntries[cacheEntry].m_VertexAttribs.m_Attributes3,
            sizeof(glm::vec3) * m_pRenderEngine->m_ShaderMetadata.m_NumVec3Attributes);

        memcpy(
            pTempVertexAttrib->m_Attributes4,
            m_VertexCacheEntries[cacheEntry].m_VertexAttribs.m_Attributes4,
            sizeof(glm::vec4) * m_pRenderEngine->m_ShaderMetadata.m_NumVec4Attributes);
    }

    void PipelineThread::CacheVertexData(uint32_t vertexIdx, const glm::vec4& vClip, const tyler::VertexAttributes& tempVertexAttrib)
    {
        // Check if VS$ cache has space and if so, append new vertex data
        if (m_NumVertexCacheEntries < g_scVertexShaderCacheSize)
        {
            m_CachedVertexIndices[m_NumVertexCacheEntries] = vertexIdx;

            m_VertexCacheEntries[m_NumVertexCacheEntries].m_ClipPos = vClip;
            m_VertexCacheEntries[m_NumVertexCacheEntries].m_VertexAttribs = tempVertexAttrib;

            ++m_NumVertexCacheEntries;
        }
    }

    bool PipelineThread::PerformVertexCacheLookup(uint32_t vertexIdx, uint32_t* pCachedIdx)
    {
        // Go through all cached entries
        for (uint32_t idx = 0; idx < m_NumVertexCacheEntries; idx++)
        {
            if (m_CachedVertexIndices[idx] == vertexIdx)
            {
                // Vertex is found in VS$, just return its entry index within the cache
                LOG("Vertex %d found in the VS$\n", vertexIdx);

                *pCachedIdx = idx;
                return true;
            }
        }

        return false;
    }

    bool PipelineThread::ExecuteFullTriangleClipping(uint32_t primIdx, const glm::vec4& v0Clip, const glm::vec4& v1Clip, const glm::vec4& v2Clip)
    {
        if constexpr (g_scFullTriangleClippingEnabled)
        {
            // Clip-space positions are to be bounded by:
            // -w < x < w   -> LEFT/RIGHT
            // -w < y < w   -> TOP/BOTTOM
            //  0 < z < w   -> NEAR/FAR
            // However, we will only clip primitives that are *completely* outside of any of clipping planes.
            // This means that, triangles intersecting view frustum  are passed as-is, to be rasterized as usual.
            // Because we're utilizing homogeneous rasterization, we don't need to do explcit line-clipping here.

            // Clip against w+x=0 left plane
            bool allOutsideLeftPlane =
                (v0Clip.x < -v0Clip.w) &&
                (v1Clip.x < -v1Clip.w) &&
                (v2Clip.x < -v2Clip.w);

            // Clip against w-x=0 right plane
            bool allOutsideRightPlane =
                (v0Clip.x > v0Clip.w) &&
                (v1Clip.x > v1Clip.w) &&
                (v2Clip.x > v2Clip.w);

            // Clip against w+y top plane
            bool allOutsideBottomPlane =
                (v0Clip.y < -v0Clip.w) &&
                (v1Clip.y < -v1Clip.w) &&
                (v2Clip.y < -v2Clip.w);

            // Clip against w-y bottom plane
            bool allOutsideTopPlane =
                (v0Clip.y > v0Clip.w) &&
                (v1Clip.y > v1Clip.w) &&
                (v2Clip.y > v2Clip.w);

            // Clip against 0<z near plane
            bool allOutsideNearPlane =
                (v0Clip.z < 0.f) &&
                (v1Clip.z < 0.f) &&
                (v2Clip.z < 0.f);

            // Clip against z>w far plane
            bool allOutsideFarPlane =
                (v0Clip.z > v0Clip.w) &&
                (v1Clip.z > v1Clip.w) &&
                (v2Clip.z > v2Clip.w);

            if (allOutsideLeftPlane ||
                allOutsideRightPlane ||
                allOutsideBottomPlane ||
                allOutsideTopPlane ||
                allOutsideNearPlane ||
                allOutsideFarPlane)
            {
                // TRIVIALREJECT case

                // Primitive completely outside of one of the clip planes, discard it
                return false;
            }
            else
            {
                // MUSTCLIP or TRIVIALACCEPT

                // Compute bounding box

                float width = static_cast<float>(m_pRenderEngine->m_Framebuffer.m_Width);
                float height = static_cast<float>(m_pRenderEngine->m_Framebuffer.m_Height);

                Rect2D bbox;
                ComputeBoundingBox(v0Clip, v1Clip, v2Clip, width, height, &bbox);

                if ((bbox.m_MinX >= width) ||
                    (bbox.m_MaxX < 0.f) ||
                    (bbox.m_MinY >= height) ||
                    (bbox.m_MaxY < 0.f))
                {
                    // If tri's bbox exceeds screen bounds, discard it
                    return false;
                }
                else
                {
                    // Clamp bbox to screen bounds
                    bbox.m_MinX = glm::max(0.f, bbox.m_MinX);
                    bbox.m_MaxX = glm::min(width, bbox.m_MaxX);
                    bbox.m_MinY = glm::max(0.f, bbox.m_MinY);
                    bbox.m_MaxY = glm::min(height, bbox.m_MaxY);

                    // Cache bbox of the primitive
                    m_pRenderEngine->m_SetupBuffers.m_pPrimBBoxes[primIdx] = bbox;

                    // Primitive is partially or completely inside view frustum, but we don't check for this
                    // so we must rasterize it further
                    return true;
                }
            }
        }
        else
        {
            // FT clipping disabled

            // Compute bounding box

            float width = static_cast<float>(m_pRenderEngine->m_Framebuffer.m_Width);
            float height = static_cast<float>(m_pRenderEngine->m_Framebuffer.m_Height);

            Rect2D bbox;
            ComputeBoundingBox(v0Clip, v1Clip, v2Clip, width, height, &bbox);

            if ((bbox.m_MinX >= width) ||
                (bbox.m_MaxX < 0.f) ||
                (bbox.m_MinY >= height) ||
                (bbox.m_MaxY < 0.f))
            {
                // If tri's bbox exceeds screen bounds, discard it
                return false;
            }
            else
            {
                // Clamp bbox to screen bounds
                bbox.m_MinX = glm::max(0.f, bbox.m_MinX);
                bbox.m_MaxX = glm::min(width, bbox.m_MaxX);
                bbox.m_MinY = glm::max(0.f, bbox.m_MinY);
                bbox.m_MaxY = glm::min(height, bbox.m_MaxY);

                // Cache bbox of the primitive
                m_pRenderEngine->m_SetupBuffers.m_pPrimBBoxes[primIdx] = bbox;

// No clipping applied
return true;
            }
        }
    }

    bool PipelineThread::ExecuteTriangleSetupAndCull(uint32_t primIdx, const glm::vec4& v0Clip, const glm::vec4& v1Clip, const glm::vec4& v2Clip)
    {
        //TODO: Cull degenerate and back-facing primitives if culling is enabled

        // Transform a given vertex in clip-space [-w,w] to device-space homogeneous coordinates [0, {w|h}]
#define TO_HOMOGEN(clipPos, width, height) glm::vec4((width * (clipPos.x + clipPos.w) * 0.5f), (height * (clipPos.y + clipPos.w) * 0.5f), clipPos.z, clipPos.w)

        float fbWidth = static_cast<float>(m_pRenderEngine->m_Framebuffer.m_Width);
        float fbHeight = static_cast<float>(m_pRenderEngine->m_Framebuffer.m_Height);

        // First, transform clip-space (x, y, z, w) vertices to device-space 2D homogeneous coordinates (x, y, w)
        const glm::vec4 v0Homogen = TO_HOMOGEN(v0Clip, fbWidth, fbHeight);
        const glm::vec4 v1Homogen = TO_HOMOGEN(v1Clip, fbWidth, fbHeight);
        const glm::vec4 v2Homogen = TO_HOMOGEN(v2Clip, fbWidth, fbHeight);

        // To calculate EE coefficients, we need to set up a "vertex matrix" and invert it
        // M = |  x0  x1  x2  |
        //     |  y0  y1  y2  |
        //     |  w0  w1  w2  |

        // Alternatively, we can rely on the following relation between an inverse and adjoint of a matrix: inv(M) = adj(M)/det(M)
        // Since we use homogeneous coordinates, it's sufficient to only compute adjoint matrix:
        // A = |  a0  b0  c0  |
        //     |  a1  b1  c1  |
        //     |  a2  b2  c2  |

        float a0 = (v2Homogen.y * v1Homogen.w) - (v1Homogen.y * v2Homogen.w);
        float a1 = (v0Homogen.y * v2Homogen.w) - (v2Homogen.y * v0Homogen.w);
        float a2 = (v1Homogen.y * v0Homogen.w) - (v0Homogen.y * v1Homogen.w);

        float b0 = (v1Homogen.x * v2Homogen.w) - (v2Homogen.x * v1Homogen.w);
        float b1 = (v2Homogen.x * v0Homogen.w) - (v0Homogen.x * v2Homogen.w);
        float b2 = (v0Homogen.x * v1Homogen.w) - (v1Homogen.x * v0Homogen.w);

        float c0 = (v2Homogen.x * v1Homogen.y) - (v1Homogen.x * v2Homogen.y);
        float c1 = (v0Homogen.x * v2Homogen.y) - (v2Homogen.x * v0Homogen.y);
        float c2 = (v1Homogen.x * v0Homogen.y) - (v0Homogen.x * v1Homogen.y);

        // Additionally,
        // det(M) == 0 -> degenerate/zero-area triangle
        // det(M) < 0  -> back-facing triangle
        float detM = (c0 * v0Homogen.w) + (c1 * v1Homogen.w) + (c2 * v2Homogen.w);

        //TODO: Proper culling!
        //TODO: If to render back-facing tris, invert the signs of elements of adj(M)
        if (detM > 0.f)
        {
            // Triangle not culled, assign computed EE coefficients for given primitive
            m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * primIdx + 0] = { a0, b0, c0 };
            m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * primIdx + 1] = { a1, b1, c1 };
            m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * primIdx + 2] = { a2, b2, c2 };

            // Store clip-space Z interpolation deltas in the setup buffer that will be used for perspective-correct interpolation of Z
            m_pRenderEngine->m_SetupBuffers.m_pInterpolatedZValues[primIdx] = { (v0Clip.z - v2Clip.z), (v1Clip.z - v2Clip.z), v2Clip.z };

            return true;
        }
        else
        {
            // Triangle culled, do nothing else
            return false;
        }
    }

    void PipelineThread::ExecuteBinner(uint32_t primIdx, const glm::vec4& v0Clip, const glm::vec4& v1Clip, const glm::vec4& v2Clip)
    {
        LOG("Thread %d binning prim %d\n", m_ThreadIdx, primIdx);

        // Binning in progress now
        m_CurrentState.store(ThreadStatus::DRAWCALL_BINNING, std::memory_order_relaxed);

        float fbWidth = static_cast<float>(m_pRenderEngine->m_Framebuffer.m_Width);
        float fbHeight = static_cast<float>(m_pRenderEngine->m_Framebuffer.m_Height);

        // Fetch bbox of the triangle computed during clipping
        Rect2D bbox = m_pRenderEngine->m_SetupBuffers.m_pPrimBBoxes[primIdx];

        ASSERT((bbox.m_MinX >= 0.f) && (bbox.m_MaxX >= 0.f) && (bbox.m_MinY >= 0.f) && (bbox.m_MaxY >= 0.f));
        ASSERT((bbox.m_MinX <= bbox.m_MaxX) && (bbox.m_MinY <= bbox.m_MaxY));

        // Given a tile size and frame buffer dimensions, find min/max range of the tiles that fall within bbox computed above
        // which we're going to iterate over, in order to determine if the primitive should be binned or not

        // Use floor(), min indices are inclusive
        uint32_t minTileX = static_cast<uint32_t>(glm::floor(bbox.m_MinX / m_RenderConfig.m_TileSize));
        uint32_t minTileY = static_cast<uint32_t>(glm::floor(bbox.m_MinY / m_RenderConfig.m_TileSize));

        // Use ceil(), max indices are exclusive
        uint32_t maxTileX = static_cast<uint32_t>(glm::ceil(bbox.m_MaxX / m_RenderConfig.m_TileSize));
        uint32_t maxTileY = static_cast<uint32_t>(glm::ceil(bbox.m_MaxY / m_RenderConfig.m_TileSize));

        ASSERT((minTileX <= maxTileX) && (maxTileX <= m_pRenderEngine->m_NumTilePerRow));
        ASSERT((minTileY <= maxTileY) && (maxTileY <= m_pRenderEngine->m_NumTilePerColumn));

        // Fetch edge equation coefficients computed in triangle setup
        glm::vec3 ee0 = m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * primIdx + 0];
        glm::vec3 ee1 = m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * primIdx + 1];
        glm::vec3 ee2 = m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * primIdx + 2];

        // Normalize edge functions
        ee0 = ee0 / (glm::abs(ee0.x) + glm::abs(ee0.y));
        ee1 = ee1 / (glm::abs(ee1.x) + glm::abs(ee1.y));
        ee2 = ee2 / (glm::abs(ee2.x) + glm::abs(ee2.y));

        // Indices of tile corners:
        // LL -> 0  LR -> 1
        // UL -> 2  UR -> 3

        static const glm::vec2 scTileCornerOffsets[] =
        {
            { 0.f, 0.f},                                            // LL (origin)
            { m_RenderConfig.m_TileSize, 0.f },                     // LR
            { 0.f, m_RenderConfig.m_TileSize },                     // UL
            { m_RenderConfig.m_TileSize, m_RenderConfig.m_TileSize} // UR
        };

        // (x, y) -> sample location | (a, b, c) -> edge equation coefficients
        // E(x, y) = (a * x) + (b * y) + c
        // E(x + s, y + t) = E(x, y) + (a * s) + (b * t)

        // Based on edge normal n=(a, b), set up tile TR corners for each edge
        const uint8_t edge0TRCorner = (ee0.y >= 0.f) ? ((ee0.x >= 0.f) ? 3u : 2u) : (ee0.x >= 0.f) ? 1u : 0u;
        const uint8_t edge1TRCorner = (ee1.y >= 0.f) ? ((ee1.x >= 0.f) ? 3u : 2u) : (ee1.x >= 0.f) ? 1u : 0u;
        const uint8_t edge2TRCorner = (ee2.y >= 0.f) ? ((ee2.x >= 0.f) ? 3u : 2u) : (ee2.x >= 0.f) ? 1u : 0u;

        // TA corner is the one diagonal from TR corner calculated above
        const uint8_t edge0TACorner = 3u - edge0TRCorner;
        const uint8_t edge1TACorner = 3u - edge1TRCorner;
        const uint8_t edge2TACorner = 3u - edge2TRCorner;

        // Evaluate edge function for the first tile within [minTile, maxTile] region
        // once and re-use it by stepping from it within following nested loop

        // Tile origin
        const float tilePosX = glm::min(fbWidth, static_cast<float>(minTileX * m_RenderConfig.m_TileSize));
        const float tilePosY = glm::min(fbHeight, static_cast<float>(minTileY * m_RenderConfig.m_TileSize));

        // No need to fetch tile positions from tiles, ensure the calculations match nevertheless
        ASSERT((tilePosX == m_pRenderEngine->m_TileList[m_pRenderEngine->GetGlobalTileIndex(minTileX, minTileY)].m_PosX));
        ASSERT((tilePosY == m_pRenderEngine->m_TileList[m_pRenderEngine->GetGlobalTileIndex(minTileX, minTileY)].m_PosY));

        // Evaluaate edge equation at first tile origin
        const float edgeFunc0 = ee0.z + (ee0.x * tilePosX) + (ee0.y * tilePosY);
        const float edgeFunc1 = ee1.z + (ee1.x * tilePosX) + (ee1.y * tilePosY);
        const float edgeFunc2 = ee2.z + (ee2.x * tilePosX) + (ee2.y * tilePosY);

        // Iterate over calculated range of tiles
        for (uint32_t ty = minTileY, tyy = 0; ty < maxTileY; ty++, tyy++)
        {
            for (uint32_t tx = minTileX, txx = 0; tx < maxTileX; tx++, txx++)
            {
                // (txx, tyy) = how many steps are done per dimension

                const float txxOffset = static_cast<float>(txx * m_RenderConfig.m_TileSize);
                const float tyyOffset = static_cast<float>(tyy * m_RenderConfig.m_TileSize);

                // Using EE coefficients calculated in TriangleSetup stage and positive half-space tests, determine one of three cases possible for each tile:
                // 1) TrivialReject -- tile within tri's bbox does not intersect tri -> move on
                // 2) TrivialAccept -- tile within tri's bbox is completely within tri -> emit a full-tile coverage mask
                // 3) Overlap       -- tile within tri's bbox intersects tri -> bin the triangle to given tile for further rasterization where block/pixel-level coverage masks will be emitted

                // Step from edge function computed above for the first tile in bbox
                float edgeFuncTR0 = (edgeFunc0 + (ee0.x * (scTileCornerOffsets[edge0TRCorner].x + txxOffset)) + (ee0.y * (scTileCornerOffsets[edge0TRCorner].y + tyyOffset)));
                float edgeFuncTR1 = (edgeFunc1 + (ee1.x * (scTileCornerOffsets[edge1TRCorner].x + txxOffset)) + (ee1.y * (scTileCornerOffsets[edge1TRCorner].y + tyyOffset)));
                float edgeFuncTR2 = (edgeFunc2 + (ee2.x * (scTileCornerOffsets[edge2TRCorner].x + txxOffset)) + (ee2.y * (scTileCornerOffsets[edge2TRCorner].y + tyyOffset)));

                // If TR corner of the tile is outside of an edge, reject whole tile
                bool TRForEdge0 = (edgeFuncTR0 < 0.f);
                bool TRForEdge1 = (edgeFuncTR1 < 0.f);
                bool TRForEdge2 = (edgeFuncTR2 < 0.f);
                if (TRForEdge0 || TRForEdge1 || TRForEdge2)
                {
                    LOG("Tile %d TR'd by thread %d\n", m_pRenderEngine->GetGlobalTileIndex(tx, ty), m_ThreadIdx);

                    // TrivialReject
                    // Tile is completely outside of one or more edges
                    continue;
                }
                else
                {
                    // Tile is partially or completely inside one or more edges, do TrivialAccept tests first

                    // Compute edge functions at TA corners based on edge function at first tile origin
                    float edgeFuncTA0 = (edgeFunc0 + (ee0.x * (scTileCornerOffsets[edge0TACorner].x + txxOffset)) + (ee0.y * (scTileCornerOffsets[edge0TACorner].y + tyyOffset)));
                    float edgeFuncTA1 = (edgeFunc1 + (ee1.x * (scTileCornerOffsets[edge1TACorner].x + txxOffset)) + (ee1.y * (scTileCornerOffsets[edge1TACorner].y + tyyOffset)));
                    float edgeFuncTA2 = (edgeFunc2 + (ee2.x * (scTileCornerOffsets[edge2TACorner].x + txxOffset)) + (ee2.y * (scTileCornerOffsets[edge2TACorner].y + tyyOffset)));

                    bool TAForEdge0 = (edgeFuncTA0 >= 0.f);
                    bool TAForEdge1 = (edgeFuncTA1 >= 0.f);
                    bool TAForEdge2 = (edgeFuncTA2 >= 0.f);
                    if (TAForEdge0 && TAForEdge1 && TAForEdge2)
                    {
                        // TrivialAccept
                        // Tile is completely inside of the triangle, no further rasterization is needed,
                        // whole tile will be fragment-shaded!

                        LOG("Tile %d TA'd by thread %d\n", m_pRenderEngine->GetGlobalTileIndex(tx, ty), m_ThreadIdx);

                        // Append tile to the rasterizer queue
                        m_pRenderEngine->EnqueueTileForRasterization(m_pRenderEngine->GetGlobalTileIndex(tx, ty));

                        // Emit full-tile coverage mask
                        CoverageMask mask;
                        mask.m_SampleX = static_cast<uint32_t>(tilePosX + txxOffset); // Based off of first tile position calculated above
                        mask.m_SampleY = static_cast<uint32_t>(tilePosY + tyyOffset); // Based off of first tile position calculated above
                        mask.m_PrimIdx = primIdx;
                        mask.m_Type = CoverageMaskType::TILE;

                        m_pRenderEngine->AppendCoverageMask(
                            m_ThreadIdx,
                            m_pRenderEngine->GetGlobalTileIndex(tx, ty),
                            mask);
                    }
                    else
                    {
                        LOG("Tile %d binned by thread %d\n", m_pRenderEngine->GetGlobalTileIndex(tx, ty), m_ThreadIdx);

                        // Overlap
                        // Tile is partially covered by the triangle, bin the triangle for the tile
                        m_pRenderEngine->BinPrimitiveForTile(
                            m_ThreadIdx,
                            m_pRenderEngine->GetGlobalTileIndex(tx, ty),
                            primIdx);
                    }
                }
            }
        }
    }

    void PipelineThread::ExecuteRasterizer()
    {
        // Request next (global) index of the tile to be rasterized at block level from RenderEngine
        uint32_t nextTileIdx;
        while ((nextTileIdx = m_pRenderEngine->m_RasterizerQueue.FetchNextTileIndex()) != g_scInvalidTileIndex)
        {
            LOG("Thread %d rasterizing tile %d\n", m_ThreadIdx, nextTileIdx);

            ASSERT(nextTileIdx < (m_pRenderEngine->m_NumTilePerRow * m_pRenderEngine->m_NumTilePerColumn));

            // Grabbed next tile from the queue, scan through its per-thread bins to rasterize the primitives
            ASSERT(m_pRenderEngine->m_BinList[nextTileIdx].size() == m_RenderConfig.m_NumPipelineThreads);

            // Tile must have been appended to the rasterizer queue, otherwise binning was incorrectly done for primitive!
            ASSERT(m_pRenderEngine->m_TileList[nextTileIdx].m_IsTileQueued.test_and_set());

            // Tile origin
            const float tilePosX = m_pRenderEngine->m_TileList[nextTileIdx].m_PosX;
            const float tilePosY = m_pRenderEngine->m_TileList[nextTileIdx].m_PosY;

            // Go through all per-thread bins in-order
            for (uint32_t i = 0; i < m_RenderConfig.m_NumPipelineThreads; i++)
            {
                // If a tile was trivially accepted, its bin will be empty
                const std::vector<uint32_t>& perThreadBin = m_pRenderEngine->m_BinList[nextTileIdx][i];

                LOG("Tile %d thread %d bin size: %d\n", nextTileIdx, i, perThreadBin.size());

                // Go through all primitives in current per-thread bin in-order
                for (uint32_t p = 0; p < perThreadBin.size(); p++)
                {
                    // Get next (global) primitive index to be rasterized
                    uint32_t primIdx = perThreadBin[p];

                    // Copy prim's bbox and clamp it to the tile edges
                    Rect2D bbox = m_pRenderEngine->m_SetupBuffers.m_pPrimBBoxes[primIdx];
                    bbox.m_MinX = glm::max(bbox.m_MinX, tilePosX);
                    bbox.m_MinY = glm::max(bbox.m_MinY, tilePosY);
                    bbox.m_MaxX = glm::min(bbox.m_MaxX, tilePosX + m_RenderConfig.m_TileSize);
                    bbox.m_MaxY = glm::min(bbox.m_MaxY, tilePosY + m_RenderConfig.m_TileSize);

                    // In case bbox is screwed up after clamping to the tile edges
                    ASSERT((bbox.m_MinX <= bbox.m_MaxX) && (bbox.m_MinY <= bbox.m_MaxY));

                    // Given a fixed 8x8 block and tile size, find min/max range of the blocks that fall within bbox computed above
                    // which we're going to iterate over, in order to determine how blocks within tile are to be rasterized

                    // Use floor(), min indices are inclusive
                    uint32_t minBlockX = static_cast<uint32_t>(glm::floor((bbox.m_MinX - tilePosX) / g_scPixelBlockSize));
                    uint32_t minBlockY = static_cast<uint32_t>(glm::floor((bbox.m_MinY - tilePosY) / g_scPixelBlockSize));
                    // Use ceil(), max indices are exclusive
                    uint32_t maxBlockX = static_cast<uint32_t>(glm::ceil((bbox.m_MaxX - tilePosX) / g_scPixelBlockSize));
                    uint32_t maxBlockY = static_cast<uint32_t>(glm::ceil((bbox.m_MaxY - tilePosY) / g_scPixelBlockSize));

                    ASSERT((minBlockX <= maxBlockX) && (maxBlockX <= m_RenderConfig.m_TileSize / g_scPixelBlockSize));
                    ASSERT((minBlockY <= maxBlockY) && (maxBlockY <= m_RenderConfig.m_TileSize / g_scPixelBlockSize));

                    // Use EE coefficients calculated in TriangleSetup again to rasterize primitive at the 8x8 block level
                    glm::vec3 ee0 = m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * primIdx + 0];
                    glm::vec3 ee1 = m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * primIdx + 1];
                    glm::vec3 ee2 = m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * primIdx + 2];

                    // Normalize edge functions
                    ee0 = ee0 / (glm::abs(ee0.x) + glm::abs(ee0.y));
                    ee1 = ee1 / (glm::abs(ee1.x) + glm::abs(ee1.y));
                    ee2 = ee2 / (glm::abs(ee2.x) + glm::abs(ee2.y));

                    static constexpr glm::vec2 scBlockCornerOffsets[] =
                    {
                        { 0.f, 0.f},                                // LL (origin)
                        { g_scPixelBlockSize, 0.f },                // LR
                        { 0.f, g_scPixelBlockSize },                // UL
                        { g_scPixelBlockSize, g_scPixelBlockSize}   // UR
                    };

                    // (x, y) -> sample location | (a, b, c) -> edge equation coefficients
                    // E(x, y) = (a * x) + (b * y) + c
                    // E(x + s, y + t) = E(x, y) + (a * s) + (b * t)

                    // Based on edge normal n=(a, b), set up block TR corners for each edge once
                    const uint8_t edge0TRCorner = (ee0.y >= 0.f) ? ((ee0.x >= 0.f) ? 3u : 2u) : (ee0.x >= 0.f) ? 1u : 0u;
                    const uint8_t edge1TRCorner = (ee1.y >= 0.f) ? ((ee1.x >= 0.f) ? 3u : 2u) : (ee1.x >= 0.f) ? 1u : 0u;
                    const uint8_t edge2TRCorner = (ee2.y >= 0.f) ? ((ee2.x >= 0.f) ? 3u : 2u) : (ee2.x >= 0.f) ? 1u : 0u;

                    const uint8_t edge0TACorner = 3u - edge0TRCorner;
                    const uint8_t edge1TACorner = 3u - edge1TRCorner;
                    const uint8_t edge2TACorner = 3u - edge2TRCorner;

                    // Evaluate edge function for the first block within [minBlock, maxBlock] region
                    // once and re-use it by stepping from it within following nested loop

                    const float firstBlockWithinBBoxX = tilePosX + minBlockX * g_scPixelBlockSize;
                    const float firstBlockWithinBBoxY = tilePosY + minBlockY * g_scPixelBlockSize;

                    // Evaluate edge equation at first block origin
                    const float edgeFunc0 = ee0.z + (ee0.x * firstBlockWithinBBoxX) + (ee0.y * firstBlockWithinBBoxY);
                    const float edgeFunc1 = ee1.z + (ee1.x * firstBlockWithinBBoxX) + (ee1.y * firstBlockWithinBBoxY);
                    const float edgeFunc2 = ee2.z + (ee2.x * firstBlockWithinBBoxX) + (ee2.y * firstBlockWithinBBoxY);

                    // Iterate over calculated range of blocks within the tile
                    for (uint32_t by = minBlockY, byy = 0; by < maxBlockY; by++, byy++)
                    {
                        for (uint32_t bx = minBlockX, bxx = 0; bx < maxBlockX; bx++, bxx++)
                        {
                            // (bxx, byy) = How many steps are done per dimension

                            // Using EE coefficients calculated in TriangleSetup stage and positive half-space tests, determine one of three cases possible for each block:
                            // 1) TrivialReject -- block within tri's bbox does not intersect tri -> move on
                            // 2) TrivialAccept -- block within tri's bbox is completely within tri -> emit a full-block coverage mask
                            // 3) Overlap       -- block within tri's bbox intersects tri -> descend into block level to emit coverage masks at pixel granularity

                            const float bxxOffset = static_cast<float>(bxx * g_scPixelBlockSize);
                            const float byyOffset = static_cast<float>(byy * g_scPixelBlockSize);

                            // Step down from edge function computed above for the first block in bbox
                            float edgeFuncTR0 = (edgeFunc0 + (ee0.x * (scBlockCornerOffsets[edge0TRCorner].x + bxxOffset)) + (ee0.y * (scBlockCornerOffsets[edge0TRCorner].y + byyOffset)));
                            float edgeFuncTR1 = (edgeFunc1 + (ee1.x * (scBlockCornerOffsets[edge1TRCorner].x + bxxOffset)) + (ee1.y * (scBlockCornerOffsets[edge1TRCorner].y + byyOffset)));
                            float edgeFuncTR2 = (edgeFunc2 + (ee2.x * (scBlockCornerOffsets[edge2TRCorner].x + bxxOffset)) + (ee2.y * (scBlockCornerOffsets[edge2TRCorner].y + byyOffset)));

                            // If TR corner of the block is outside of an edge, reject whole block
                            bool TRForEdge0 = (edgeFuncTR0 < 0.f);
                            bool TRForEdge1 = (edgeFuncTR1 < 0.f);
                            bool TRForEdge2 = (edgeFuncTR2 < 0.f);
                            if (TRForEdge0 || TRForEdge1 || TRForEdge2)
                            {
                                LOG("Tile %d block (%d, %d) TR'd by thread %d\n", nextTileIdx, bx, by, m_ThreadIdx);

                                // TrivialReject
                                // Block is completely outside of one or more edges
                                continue;
                            }
                            else
                            {
                                // Block is partially or completely inside one or more edges, do TrivialAccept tests first

                                float edgeFuncTA0 = (edgeFunc0 + (ee0.x * (scBlockCornerOffsets[edge0TACorner].x + bxxOffset)) + (ee0.y * (scBlockCornerOffsets[edge0TACorner].y + byyOffset)));
                                float edgeFuncTA1 = (edgeFunc1 + (ee1.x * (scBlockCornerOffsets[edge1TACorner].x + bxxOffset)) + (ee1.y * (scBlockCornerOffsets[edge1TACorner].y + byyOffset)));
                                float edgeFuncTA2 = (edgeFunc2 + (ee2.x * (scBlockCornerOffsets[edge2TACorner].x + bxxOffset)) + (ee2.y * (scBlockCornerOffsets[edge2TACorner].y + byyOffset)));

                                // Compute edge functions at TA corners by stepping from TR values already calculated above
                                bool TAForEdge0 = (edgeFuncTA0 >= 0.f);
                                bool TAForEdge1 = (edgeFuncTA1 >= 0.f);
                                bool TAForEdge2 = (edgeFuncTA2 >= 0.f);
                                if (TAForEdge0 && TAForEdge1 && TAForEdge2)
                                {
                                    // TrivialAccept
                                    // Block is completely inside of the triangle, emit a full-block coverage mask

                                    LOG("Tile %d block (%d, %d) TA'd by thread %d\n", nextTileIdx, bx, by, m_ThreadIdx);

                                    CoverageMask mask;
                                    mask.m_SampleX = static_cast<uint32_t>(firstBlockWithinBBoxX + bxxOffset); // Based off of first block position calculated above
                                    mask.m_SampleY = static_cast<uint32_t>(firstBlockWithinBBoxY + byyOffset); // Based off of first block position calculated above
                                    mask.m_PrimIdx = primIdx;
                                    mask.m_Type = CoverageMaskType::BLOCK;

                                    m_pRenderEngine->AppendCoverageMask(
                                        m_ThreadIdx,
                                        nextTileIdx,
                                        mask);
                                }
                                else
                                {
                                    // Overlap
                                    // Block is partially covered by the triangle, descend into pixel level and perform edge tests

                                    LOG("Tile %d block (%d, %d) overlapping tests by thread %d\n", nextTileIdx, bx, by, m_ThreadIdx);

                                    // Position of the block that we're testing at pixel level
                                    float blockPosX = (firstBlockWithinBBoxX + bxxOffset);
                                    float blockPosY = (firstBlockWithinBBoxY + byyOffset);

                                    // Compute E(x, y) = (x * a) + (y * b) c at block origin once
                                    __m128 sseEdge0FuncAtBlockOrigin = _mm_set1_ps(ee0.z + (ee0.x * blockPosX) + (ee0.y * blockPosY));
                                    __m128 sseEdge1FuncAtBlockOrigin = _mm_set1_ps(ee1.z + (ee1.x * blockPosX) + (ee1.y * blockPosY));
                                    __m128 sseEdge2FuncAtBlockOrigin = _mm_set1_ps(ee2.z + (ee2.x * blockPosX) + (ee2.y * blockPosY));

                                    // Store edge 0 equation coefficients
                                    __m128 sseEdge0A4 = _mm_set_ps1(ee0.x);
                                    __m128 sseEdge0B4 = _mm_set_ps1(ee0.y);

                                    // Store edge 1 equation coefficients
                                    __m128 sseEdge1A4 = _mm_set_ps1(ee1.x);
                                    __m128 sseEdge1B4 = _mm_set_ps1(ee1.y);

                                    // Store edge 2 equation coefficients
                                    __m128 sseEdge2A4 = _mm_set_ps1(ee2.x);
                                    __m128 sseEdge2B4 = _mm_set_ps1(ee2.y);

                                    // Generate masks used for tie-breaking rules (not to double-shade along shared edges)
                                    __m128 sseEdge0A4PositiveOrB4NonNegativeA4Zero = _mm_or_ps(_mm_cmpgt_ps(sseEdge0A4, _mm_setzero_ps()),
                                        _mm_and_ps(_mm_cmpge_ps(sseEdge0B4, _mm_setzero_ps()), _mm_cmpeq_ps(sseEdge0A4, _mm_setzero_ps())));

                                    __m128 sseEdge1A4PositiveOrB4NonNegativeA4Zero = _mm_or_ps(_mm_cmpgt_ps(sseEdge1A4, _mm_setzero_ps()),
                                        _mm_and_ps(_mm_cmpge_ps(sseEdge1B4, _mm_setzero_ps()), _mm_cmpeq_ps(sseEdge1A4, _mm_setzero_ps())));

                                    __m128 sseEdge2A4PositiveOrB4NonNegativeA4Zero = _mm_or_ps(_mm_cmpgt_ps(sseEdge2A4, _mm_setzero_ps()),
                                        _mm_and_ps(_mm_cmpge_ps(sseEdge2B4, _mm_setzero_ps()), _mm_cmpeq_ps(sseEdge2A4, _mm_setzero_ps())));

                                    for (uint32_t py = 0; py < g_scPixelBlockSize; py++)
                                    {
                                        // Store Y positions in current row (all samples on the same row has the same Y position)
                                        __m128 sseY4 = _mm_set_ps1(py + 0.5f);

                                        for (uint32_t px = 0; px < g_scNumEdgeTestsPerRow; px++)
                                        {
                                            // E(x, y) = (x * a) + (y * b) + c
                                            // E(x + s, y + t) = E(x, y) + s * a + t * b

#ifdef _DEBUG
                                            int32_t debugMaskScalar = 0;
                                            {
                                                // Debug for SSE edge tests

                                                float edge0FuncAtBlockOrigin = ee0.z + (ee0.x * blockPosX) + (ee0.y * blockPosY);
                                                float edge1FuncAtBlockOrigin = ee1.z + (ee1.x * blockPosX) + (ee1.y * blockPosY);
                                                float edge2FuncAtBlockOrigin = ee2.z + (ee2.x * blockPosX) + (ee2.y * blockPosY);

                                                // 4 Sample locations
                                                glm::vec2 sample0 = { g_scSIMDWidth * px + 0.5f, py + 0.5f };
                                                glm::vec2 sample1 = { g_scSIMDWidth * px + 1.5f, py + 0.5f };
                                                glm::vec2 sample2 = { g_scSIMDWidth * px + 2.5f, py + 0.5f };
                                                glm::vec2 sample3 = { g_scSIMDWidth * px + 3.5f, py + 0.5f };

                                                bool inside0 =
                                                    EvaluateEdgeFunctionIncremental(ee0, sample0, edge0FuncAtBlockOrigin) &&
                                                    EvaluateEdgeFunctionIncremental(ee1, sample0, edge1FuncAtBlockOrigin) &&
                                                    EvaluateEdgeFunctionIncremental(ee2, sample0, edge2FuncAtBlockOrigin);

                                                bool inside1 =
                                                    EvaluateEdgeFunctionIncremental(ee0, sample1, edge0FuncAtBlockOrigin) &&
                                                    EvaluateEdgeFunctionIncremental(ee1, sample1, edge1FuncAtBlockOrigin) &&
                                                    EvaluateEdgeFunctionIncremental(ee2, sample1, edge2FuncAtBlockOrigin);

                                                bool inside2 =
                                                    EvaluateEdgeFunctionIncremental(ee0, sample2, edge0FuncAtBlockOrigin) &&
                                                    EvaluateEdgeFunctionIncremental(ee1, sample2, edge1FuncAtBlockOrigin) &&
                                                    EvaluateEdgeFunctionIncremental(ee2, sample2, edge2FuncAtBlockOrigin);

                                                bool inside3 =
                                                    EvaluateEdgeFunctionIncremental(ee0, sample3, edge0FuncAtBlockOrigin) &&
                                                    EvaluateEdgeFunctionIncremental(ee1, sample3, edge1FuncAtBlockOrigin) &&
                                                    EvaluateEdgeFunctionIncremental(ee2, sample3, edge2FuncAtBlockOrigin);

                                                if (inside0) debugMaskScalar |= g_scQuadMask0;
                                                if (inside1) debugMaskScalar |= g_scQuadMask1;
                                                if (inside2) debugMaskScalar |= g_scQuadMask2;
                                                if (inside3) debugMaskScalar |= g_scQuadMask3;
                                            }
#endif

                                            // Store X positions of 4 consecutive samples
                                            __m128 sseX4 = _mm_setr_ps(
                                                g_scSIMDWidth * px + 0.5f,
                                                g_scSIMDWidth * px + 1.5f,
                                                g_scSIMDWidth * px + 2.5f,
                                                g_scSIMDWidth * px + 3.5f);

                                            // a * s
                                            __m128 sseEdge0TermA = _mm_mul_ps(sseEdge0A4, sseX4);
                                            __m128 sseEdge1TermA = _mm_mul_ps(sseEdge1A4, sseX4);
                                            __m128 sseEdge2TermA = _mm_mul_ps(sseEdge2A4, sseX4);

                                            // b * t
                                            __m128 sseEdge0TermB = _mm_mul_ps(sseEdge0B4, sseY4);
                                            __m128 sseEdge1TermB = _mm_mul_ps(sseEdge1B4, sseY4);
                                            __m128 sseEdge2TermB = _mm_mul_ps(sseEdge2B4, sseY4);

                                            // E(x+s, y+t) = E(x,y) + a*s + t*b
                                            __m128 sseEdgeFunc0 = _mm_add_ps(sseEdge0FuncAtBlockOrigin, _mm_add_ps(sseEdge0TermA, sseEdge0TermB));
                                            __m128 sseEdgeFunc1 = _mm_add_ps(sseEdge1FuncAtBlockOrigin, _mm_add_ps(sseEdge1TermA, sseEdge1TermB));
                                            __m128 sseEdgeFunc2 = _mm_add_ps(sseEdge2FuncAtBlockOrigin, _mm_add_ps(sseEdge2TermA, sseEdge2TermB));

#ifdef EDGE_TEST_SHARED_EDGES
                                            //E(x, y) =
                                            //    E(x, y) > 0
                                            //        ||
                                            //    !E(x, y) < 0 && (a > 0 || (a = 0 && b >= 0))
                                            //

                                            // Edge 0 test
                                            __m128 sseEdge0Positive = _mm_cmpgt_ps(sseEdgeFunc0, _mm_setzero_ps());
                                            __m128 sseEdge0Negative = _mm_cmplt_ps(sseEdgeFunc0, _mm_setzero_ps());
                                            __m128 sseEdge0FuncMask = _mm_or_ps(sseEdge0Positive,
                                                _mm_andnot_ps(sseEdge0Negative, sseEdge0A4PositiveOrB4NonNegativeA4Zero));

                                            // Edge 1 test
                                            __m128 sseEdge1Positive = _mm_cmpgt_ps(sseEdgeFunc1, _mm_setzero_ps());
                                            __m128 sseEdge1Negative = _mm_cmplt_ps(sseEdgeFunc1, _mm_setzero_ps());
                                            __m128 sseEdge1FuncMask = _mm_or_ps(sseEdge1Positive,
                                                _mm_andnot_ps(sseEdge1Negative, sseEdge1A4PositiveOrB4NonNegativeA4Zero));

                                            // Edge 2 test
                                            __m128 sseEdge2Positive = _mm_cmpgt_ps(sseEdgeFunc2, _mm_setzero_ps());
                                            __m128 sseEdge2Negative = _mm_cmplt_ps(sseEdgeFunc2, _mm_setzero_ps());
                                            __m128 sseEdge2FuncMask = _mm_or_ps(sseEdge2Positive,
                                                _mm_andnot_ps(sseEdge2Negative, sseEdge2A4PositiveOrB4NonNegativeA4Zero));
#else
                                            __m128 sseEdge0FuncMask = _mm_cmpge_ps(sseEdgeFunc0, _mm_setzero_ps());
                                            __m128 sseEdge1FuncMask = _mm_cmpge_ps(sseEdgeFunc1, _mm_setzero_ps());
                                            __m128 sseEdge2FuncMask = _mm_cmpge_ps(sseEdgeFunc2, _mm_setzero_ps());
#endif
                                            // Combine resulting masks of all three edges
                                            __m128 sseEdgeFuncResult = _mm_and_ps(sseEdge0FuncMask,
                                                _mm_and_ps(sseEdge1FuncMask, sseEdge2FuncMask));

                                            uint16_t maskInt = static_cast<uint16_t>(_mm_movemask_ps(sseEdgeFuncResult));

#ifdef _DEBUG
                                            // Edge functions were computed incorrectly if that fires!!!
                                            ASSERT(maskInt == debugMaskScalar);
#endif

                                            // If at least one sample is visible, emit coverage mask for the tile
                                            if (maskInt != 0x0)
                                            {
                                                // Quad mask points to the first sample
                                                CoverageMask mask;
                                                mask.m_SampleX = static_cast<uint32_t>(blockPosX + (g_scSIMDWidth * px));
                                                mask.m_SampleY = static_cast<uint32_t>(blockPosY + py);
                                                mask.m_PrimIdx = primIdx;
                                                mask.m_Type = CoverageMaskType::QUAD;
                                                mask.m_QuadMask = maskInt;

                                                // Emit a quad mask
                                                m_pRenderEngine->AppendCoverageMask(m_ThreadIdx, nextTileIdx, mask);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Allocate space for more coverage masks, if needed
                    m_pRenderEngine->ResizeCoverageMaskBuffer(m_ThreadIdx, nextTileIdx);
                }
            }
        }
    }

    void PipelineThread::ExecuteFragmentShader()
    {
        uint32_t nextTileIdx;
        while ((nextTileIdx = m_pRenderEngine->m_RasterizerQueue.RemoveTileIndex()) != g_scInvalidTileIndex)
        {
            LOG("Thread %d fragment shader for tile %d\n", m_ThreadIdx, nextTileIdx);

            ASSERT(nextTileIdx < (m_pRenderEngine->m_NumTilePerRow * m_pRenderEngine->m_NumTilePerColumn));

            // Fragment-shade visible samples consuming coverage masks emitted previously by the rasterizer stage

            // Get per-thread coverage mask and process them in order
            for (uint32_t i = 0; i < m_RenderConfig.m_NumPipelineThreads; i++)
            {
                CoverageMaskBuffer* pCoverageMaskBuffer = m_pRenderEngine->m_CoverageMasks[nextTileIdx][i];
                ASSERT(pCoverageMaskBuffer != nullptr);
                ASSERT(pCoverageMaskBuffer->m_NumAllocations > 0);

                for (uint32_t numAlloc = 0; numAlloc < pCoverageMaskBuffer->m_NumAllocations; numAlloc++)
                {
                    auto& currentSlot = pCoverageMaskBuffer->m_AllocationList[numAlloc];

                    for (uint32_t numMask = 0; numMask < currentSlot.m_AllocationCount; numMask++)
                    {
                        ASSERT(pCoverageMaskBuffer->m_AllocationList[numAlloc].m_pData != nullptr);

                        CoverageMask* pMask = &currentSlot.m_pData[numMask];
                        switch (pMask->m_Type)
                        {
                        case CoverageMaskType::TILE:
                            LOG("Thread %d fragment-shading tile %d\n", m_ThreadIdx, nextTileIdx);
                            FragmentShadeTile(pMask->m_SampleX, pMask->m_SampleY, pMask->m_PrimIdx);
                            break;
                        case CoverageMaskType::BLOCK:
                            LOG("Thread %d fragment-shading blocks\n", m_ThreadIdx);
                            FragmentShadeBlock(pMask->m_SampleX, pMask->m_SampleY, pMask->m_PrimIdx);
                            break;
                        case CoverageMaskType::QUAD:
                            LOG("Thread %d fragment-shading coverage masks\n", m_ThreadIdx);
                            FragmentShadeQuad(pMask);
                            break;
                        default:
                            ASSERT(false);
                            break;
                        }
                    }
                }
            }
        }
    }

    void PipelineThread::FragmentShadeTile(uint32_t tilePosX, uint32_t tilePosY, uint32_t primIdx)
    {
        static const uint32_t numBlockInTile = m_RenderConfig.m_TileSize / g_scPixelBlockSize;

        for (uint32_t py = 0; py < numBlockInTile; py++)
        {
            for (uint32_t px = 0; px < numBlockInTile; px++)
            {
                FragmentShadeBlock(tilePosX + px * g_scPixelBlockSize, tilePosY + py * g_scPixelBlockSize, primIdx);
            }
        }
    }

    void PipelineThread::FragmentShadeBlock(uint32_t blockPosX, uint32_t blockPosY, uint32_t primIdx)
    {
        FragmentShader FS = m_pRenderEngine->m_FragmentShader;
        ASSERT(FS != nullptr);

        InterpolatedAttributes interpolatedAttribs;

        // First fetch EE coefficients that will be used (in addition to edge in/out tests) for perspective-correct interpolation of vertex attributes
        const glm::vec3& ee0 = m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * primIdx + 0];
        const glm::vec3& ee1 = m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * primIdx + 1];
        const glm::vec3& ee2 = m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * primIdx + 2];

        // Store EE coefficients
        __m128 sseA4Edge0 = _mm_set_ps1(ee0.x);
        __m128 sseB4Edge0 = _mm_set_ps1(ee0.y);
        __m128 sseC4Edge0 = _mm_set_ps1(ee0.z);

        // Store edge 1 equation coefficients
        __m128 sseA4Edge1 = _mm_set_ps1(ee1.x);
        __m128 sseB4Edge1 = _mm_set_ps1(ee1.y);
        __m128 sseC4Edge1 = _mm_set_ps1(ee1.z);

        // Store edge 2 equation coefficients
        __m128 sseA4Edge2 = _mm_set_ps1(ee2.x);
        __m128 sseB4Edge2 = _mm_set_ps1(ee2.y);
        __m128 sseC4Edge2 = _mm_set_ps1(ee2.z);

        // Parameter interpolation basis functions
        __m128 ssef0XY, ssef1XY;

        // 4-sample fragment colors
        FragmentOutput fragmentOutput;

        // Loop over 8x8 pixels
        for (uint32_t py = 0; py < g_scPixelBlockSize; py++)
        {
            //TODO: Update frame/depth buffer w/ SIMD!

            for (uint32_t px = 0; px < g_scNumEdgeTestsPerRow; px++)
            {
                uint32_t sampleX = blockPosX + (g_scSIMDWidth * px);
                uint32_t sampleY = blockPosY + py;

                // Calculate basis functions f0(x,y) & f1(x,y) once
                ComputeParameterBasisFunctions(
                    sampleX,
                    sampleY,
                    sseA4Edge0,
                    sseB4Edge0,
                    sseC4Edge0,
                    sseA4Edge1,
                    sseB4Edge1,
                    sseC4Edge1,
                    sseA4Edge2,
                    sseB4Edge2,
                    sseC4Edge2,
                    &ssef0XY,
                    &ssef1XY);

                // Interpolate Z (4 samples)
                __m128 sseZInterpolated;
                InterpolateDepthValues(primIdx, ssef0XY, ssef1XY, &sseZInterpolated);

                // Load current depth buffer contents
                __m128 sseDepthCurrent = m_pRenderEngine->FetchDepthBuffer(sampleX, sampleY);

                // Perform LESS_THAN_EQUAL depth test
                __m128 sseDepthRes = _mm_cmple_ps(sseZInterpolated, sseDepthCurrent);

                // Pseudo Early-Z test
                if (_mm_movemask_ps(sseDepthRes) == 0x0)
                {
                    // No sample being processed passes depth test, skip invoking FS altogether
                    continue;
                }

                // Interpolate active vertex attributes
                InterpolateVertexAttributes(primIdx, ssef0XY, ssef1XY, &interpolatedAttribs);

                // Invoke FS and update color/depth buffer with fragment output
                FS(&interpolatedAttribs, m_pRenderEngine->m_pConstantBuffer, &fragmentOutput);

                // Write interpolated Z values
                m_pRenderEngine->UpdateDepthBuffer(sseDepthRes, sseZInterpolated, sampleX, sampleY);

                // Write fragment output
                m_pRenderEngine->UpdateColorBuffer(sseDepthRes, fragmentOutput, sampleX, sampleY);
            }
        }
    }

    void PipelineThread::FragmentShadeQuad(CoverageMask* pMask)
    {
        ASSERT(pMask != nullptr);

        FragmentShader FS = m_pRenderEngine->m_FragmentShader;
        ASSERT(FS != nullptr);

        // Vertex attributes to be interpolated and passed to FS
        InterpolatedAttributes interpolatedAttribs;

        // Parameter interpolation basis functions
        __m128 ssef0XY, ssef1XY;

        // Fetch EE coefficients that will be used (in addition to edge in/out tests) for perspective-correct interpolation of vertex attributes
        const glm::vec3& ee0 = m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * pMask->m_PrimIdx + 0];
        const glm::vec3& ee1 = m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * pMask->m_PrimIdx + 1];
        const glm::vec3& ee2 = m_pRenderEngine->m_SetupBuffers.m_pEdgeCoefficients[3 * pMask->m_PrimIdx + 2];

        // Store edge 0 equation coefficients
        __m128 sseA4Edge0 = _mm_set_ps1(ee0.x);
        __m128 sseB4Edge0 = _mm_set_ps1(ee0.y);
        __m128 sseC4Edge0 = _mm_set_ps1(ee0.z);

        // Store edge 1 equation coefficients
        __m128 sseA4Edge1 = _mm_set_ps1(ee1.x);
        __m128 sseB4Edge1 = _mm_set_ps1(ee1.y);
        __m128 sseC4Edge1 = _mm_set_ps1(ee1.z);

        // Store edge 2 equation coefficients
        __m128 sseA4Edge2 = _mm_set_ps1(ee2.x);
        __m128 sseB4Edge2 = _mm_set_ps1(ee2.y);
        __m128 sseC4Edge2 = _mm_set_ps1(ee2.z);

        // Calculate basis functions f0(x,y) & f1(x,y) once
        ComputeParameterBasisFunctions(
            pMask->m_SampleX,
            pMask->m_SampleY,
            sseA4Edge0,
            sseB4Edge0,
            sseC4Edge0,
            sseA4Edge1,
            sseB4Edge1,
            sseC4Edge1,
            sseA4Edge2,
            sseB4Edge2,
            sseC4Edge2,
            &ssef0XY,
            &ssef1XY);

        // Interpolate depth values prior to depth test
        __m128 sseZInterpolated;
        InterpolateDepthValues(pMask->m_PrimIdx, ssef0XY, ssef1XY, &sseZInterpolated);

        // Load current depth buffer contents
        __m128 sseDepthCurrent = m_pRenderEngine->FetchDepthBuffer(pMask->m_SampleX, pMask->m_SampleY);

        // Perform LESS_THAN_EQUAL depth test
        __m128 sseDepthRes = _mm_cmple_ps(sseZInterpolated, sseDepthCurrent);

        // Pseudo Early-Z test
        if (_mm_movemask_ps(sseDepthRes) == 0x0)
        {
            // No sample within current quad of fragments being processed passes depth test, skip it
            return;
        }

        // Interpolate active vertex attributes
        InterpolateVertexAttributes(pMask->m_PrimIdx, ssef0XY, ssef1XY, &interpolatedAttribs);

        // 4-sample fragment colors
        FragmentOutput fragmentOutput;

        // Invoke FS and update color/depth buffer with fragment output
        FS(&interpolatedAttribs, m_pRenderEngine->m_pConstantBuffer, &fragmentOutput);

        // Generate color mask from 4-bit int mask set during rasterization
        __m128i sseColorMask = _mm_setr_epi32(
            pMask->m_QuadMask & g_scQuadMask0,
            pMask->m_QuadMask & g_scQuadMask1,
            pMask->m_QuadMask & g_scQuadMask2,
            pMask->m_QuadMask & g_scQuadMask3);

        sseColorMask = _mm_cmpeq_epi32(sseColorMask,
            _mm_set_epi64x(0x800000004, 0x200000001));

        // AND depth mask & coverage mask for quads of fragments
        __m128 sseWriteMask = _mm_and_ps(sseDepthRes, _mm_castsi128_ps(sseColorMask));

        // Write interpolated Z values
        m_pRenderEngine->UpdateDepthBuffer(sseWriteMask, sseZInterpolated, pMask->m_SampleX, pMask->m_SampleY);

        // Write fragment output
        m_pRenderEngine->UpdateColorBuffer(sseWriteMask, fragmentOutput, pMask->m_SampleX, pMask->m_SampleY);
    }

    void PipelineThread::ComputeBoundingBox(const glm::vec4& v0Clip, const glm::vec4& v1Clip, const glm::vec4& v2Clip, float width, float height, Rect2D* pBbox) const
    {
        ASSERT(pBbox != nullptr);

        // Compute NDC vertices; confined to 2D because we don't need z here
        glm::vec2 v0NDC = glm::vec2(v0Clip.x, v0Clip.y) / v0Clip.w;
        glm::vec2 v1NDC = glm::vec2(v1Clip.x, v1Clip.y) / v1Clip.w;
        glm::vec2 v2NDC = glm::vec2(v2Clip.x, v2Clip.y) / v2Clip.w;

        // Transform NDC [-1, 1] -> RASTER [0, {width|height}]
        glm::vec2 v0Raster = { width * (v0NDC.x + 1.f) * 0.5f, height * (v0NDC.y + 1.f) * 0.5f };
        glm::vec2 v1Raster = { width * (v1NDC.x + 1.f) * 0.5f, height * (v1NDC.y + 1.f) * 0.5f };
        glm::vec2 v2Raster = { width * (v2NDC.x + 1.f) * 0.5f, height * (v2NDC.y + 1.f) * 0.5f };

        // Find min/max in X & Y
        float xmin = glm::min(v0Raster.x, glm::min(v1Raster.x, v2Raster.x));
        float xmax = glm::max(v0Raster.x, glm::max(v1Raster.x, v2Raster.x));
        float ymin = glm::min(v0Raster.y, glm::min(v1Raster.y, v2Raster.y));
        float ymax = glm::max(v0Raster.y, glm::max(v1Raster.y, v2Raster.y));

        pBbox->m_MinX = xmin;
        pBbox->m_MinY = ymin;
        pBbox->m_MaxX = xmax;
        pBbox->m_MaxY = ymax;
    }

    void PipelineThread::CalculateInterpolationCoefficients(
        uint32_t drawIDx,
        const VertexAttributes& vertexAttribs0,
        const VertexAttributes& vertexAttribs1,
        const VertexAttributes& vertexAttribs2)
    {
        // f0 + f1 + f2 = 1
        // f0 * x0 + f1 * x1 + f2 * x2 ==> f0 * (x0 - x2) + f1 * (x1 - x2) + x2

        // vec4 attributes
        for (uint32_t i = 0; i < m_pRenderEngine->m_ShaderMetadata.m_NumVec4Attributes; i++)
        {
            const glm::vec4& attrib0 = vertexAttribs0.m_Attributes4[i];
            const glm::vec4& attrib1 = vertexAttribs1.m_Attributes4[i];
            const glm::vec4& attrib2 = vertexAttribs2.m_Attributes4[i];

            // Store computed deltas in setup buffers for vec4 xyzw attributes
            m_pRenderEngine->m_SetupBuffers.m_Attribute4Deltas[i][drawIDx * 4 + 0] = glm::vec3((attrib0.x - attrib2.x), (attrib1.x - attrib2.x), attrib2.x);
            m_pRenderEngine->m_SetupBuffers.m_Attribute4Deltas[i][drawIDx * 4 + 1] = glm::vec3((attrib0.y - attrib2.y), (attrib1.y - attrib2.y), attrib2.y);
            m_pRenderEngine->m_SetupBuffers.m_Attribute4Deltas[i][drawIDx * 4 + 2] = glm::vec3((attrib0.z - attrib2.z), (attrib1.z - attrib2.z), attrib2.z);
            m_pRenderEngine->m_SetupBuffers.m_Attribute4Deltas[i][drawIDx * 4 + 3] = glm::vec3((attrib0.w - attrib2.w), (attrib1.w - attrib2.w), attrib2.w);
        }

        // vec3 attributes
        for (uint32_t i = 0; i < m_pRenderEngine->m_ShaderMetadata.m_NumVec3Attributes; i++)
        {
            const glm::vec3& attrib0 = vertexAttribs0.m_Attributes3[i];
            const glm::vec3& attrib1 = vertexAttribs1.m_Attributes3[i];
            const glm::vec3& attrib2 = vertexAttribs2.m_Attributes3[i];

            // Store computed deltas in setup buffers for vec3 xyz attributes
            m_pRenderEngine->m_SetupBuffers.m_Attribute3Deltas[i][drawIDx * 3 + 0] = glm::vec3((attrib0.x - attrib2.x), (attrib1.x - attrib2.x), attrib2.x);
            m_pRenderEngine->m_SetupBuffers.m_Attribute3Deltas[i][drawIDx * 3 + 1] = glm::vec3((attrib0.y - attrib2.y), (attrib1.y - attrib2.y), attrib2.y);
            m_pRenderEngine->m_SetupBuffers.m_Attribute3Deltas[i][drawIDx * 3 + 2] = glm::vec3((attrib0.z - attrib2.z), (attrib1.z - attrib2.z), attrib2.z);
        }

        // vec2 attributes
        for (uint32_t i = 0; i < m_pRenderEngine->m_ShaderMetadata.m_NumVec2Attributes; i++)
        {
            const glm::vec2& attrib0 = vertexAttribs0.m_Attributes2[i];
            const glm::vec2& attrib1 = vertexAttribs1.m_Attributes2[i];
            const glm::vec2& attrib2 = vertexAttribs2.m_Attributes2[i];

            // Store computed deltas in setup buffers for vec2 xy attributes
            m_pRenderEngine->m_SetupBuffers.m_Attribute2Deltas[i][drawIDx * 2 + 0] = glm::vec3((attrib0.x - attrib2.x), (attrib1.x - attrib2.x), attrib2.x);
            m_pRenderEngine->m_SetupBuffers.m_Attribute2Deltas[i][drawIDx * 2 + 1] = glm::vec3((attrib0.y - attrib2.y), (attrib1.y - attrib2.y), attrib2.y);
        }
    }

    void PipelineThread::ComputeParameterBasisFunctions(
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
        __m128* pSSEf1XY)
    {
        // R(x, y) = F0(x, y) + F1(x, y) + F2(x, y)
        // r = 1/(F0(x, y) + F1(x, y) + F2(x, y))

        //TODO: Optimize w/ incremental F(x, y) evaluations!

        // Store X positions of 4 consecutive samples
        __m128 sseX4 = _mm_setr_ps(
            sampleX + 0.5f,
            sampleX + 1.5f,
            sampleX + 2.5f,
            sampleX + 3.5f); // x x+1 x+2 x+3

        // Store Y positions of 4 samples in a row (constant)
        __m128 sseY4 = _mm_set_ps1(sampleY); // y y y y

        // Compute F0(x,y)
        __m128 sseF0XY4 = _mm_add_ps(sseC4Edge0,
            _mm_add_ps(
                _mm_mul_ps(sseY4, sseB4Edge0),
                _mm_mul_ps(sseX4, sseA4Edge0)));

        // Compute F1(x,y)
        __m128 sseF1XY4 = _mm_add_ps(sseC4Edge1,
            _mm_add_ps(
                _mm_mul_ps(sseY4, sseB4Edge1),
                _mm_mul_ps(sseX4, sseA4Edge1)));

        // Compute F2(x,y)
        __m128 sseF2XY4 = _mm_add_ps(sseC4Edge2,
            _mm_add_ps(
                _mm_mul_ps(sseY4, sseB4Edge2),
                _mm_mul_ps(sseX4, sseA4Edge2)));

        // Compute F(x,y) = F0(x,y) + F1(x,y) + F2(x,y)
        __m128 sseR4 = _mm_add_ps(sseF2XY4,
            _mm_add_ps(sseF0XY4, sseF1XY4));

        // Compute perspective correction factor
        sseR4 = _mm_rcp_ps(sseR4);

        // Assign final f0(x,y) & f1(x,y)
        *pSSEf0XY = _mm_mul_ps(sseR4, sseF0XY4);
        *pSSEf1XY = _mm_mul_ps(sseR4, sseF1XY4);

        // Basis functions f0, f1, f2 sum to 1, e.g. f0(x,y) + f1(x,y) + f2(x,y) = 1 so we'll skip computing f2(x,y) explicitly
    }

    void PipelineThread::InterpolateDepthValues(uint32_t primIdx, const __m128& ssef0XY, const __m128& ssef1XY, __m128* pZInterpolated)
    {
        // Fetch interpolation deltas computed after VS was returned
        const glm::vec3& attrib0Vec3 = m_pRenderEngine->m_SetupBuffers.m_pInterpolatedZValues[primIdx];

        // vec3::x attribute to be interpolated
        __m128 sseAttrib0 = _mm_set_ps1(attrib0Vec3.x);
        __m128 sseAttrib1 = _mm_set_ps1(attrib0Vec3.y);
        __m128 sseAttrib2 = _mm_set_ps1(attrib0Vec3.z);

        *pZInterpolated = _mm_add_ps(sseAttrib2,
            _mm_add_ps(_mm_mul_ps(sseAttrib0, ssef0XY),
                _mm_mul_ps(sseAttrib1, ssef1XY)));
    }

    void PipelineThread::InterpolateVertexAttributes(
        uint32_t primIdx,
        const __m128& ssef0XY,
        const __m128& ssef1XY,
        InterpolatedAttributes* pInterpolatedAttributes)
    {
        // vec4 xyzw attributes
        for (uint32_t i = 0; i < m_pRenderEngine->m_ShaderMetadata.m_NumVec4Attributes; i++)
        {
            // Fetch interpolation deltas computed after VS was returned
            const glm::vec3& attrib0Vec3 = m_pRenderEngine->m_SetupBuffers.m_Attribute4Deltas[i][(primIdx * 4) + 0];
            const glm::vec3& attrib1Vec3 = m_pRenderEngine->m_SetupBuffers.m_Attribute4Deltas[i][(primIdx * 4) + 1];
            const glm::vec3& attrib2Vec3 = m_pRenderEngine->m_SetupBuffers.m_Attribute4Deltas[i][(primIdx * 4) + 2];
            const glm::vec3& attrib3Vec3 = m_pRenderEngine->m_SetupBuffers.m_Attribute4Deltas[i][(primIdx * 4) + 3];

            // vec4::x attribute to be interpolated
            __m128 sseAttrib0X = _mm_set_ps1(attrib0Vec3.x);
            __m128 sseAttrib1X = _mm_set_ps1(attrib0Vec3.y);
            __m128 sseAttrib2X = _mm_set_ps1(attrib0Vec3.z);

            __m128 sseVec4AttribX = _mm_add_ps(
                _mm_mul_ps(sseAttrib0X, ssef0XY),
                _mm_add_ps(_mm_mul_ps(sseAttrib1X, ssef1XY), sseAttrib2X));

            // vec4::y attribute to be interpolated
            __m128 sseAttrib0Y = _mm_set_ps1(attrib1Vec3.x);
            __m128 sseAttrib1Y = _mm_set_ps1(attrib1Vec3.y);
            __m128 sseAttrib2Y = _mm_set_ps1(attrib1Vec3.z);

            __m128 sseVec4AttribY = _mm_add_ps(
                _mm_mul_ps(sseAttrib0Y, ssef0XY),
                _mm_add_ps(_mm_mul_ps(sseAttrib1Y, ssef1XY), sseAttrib2Y));

            // vec4::z attribute to be interpolated
            __m128 sseAttrib0Z = _mm_set_ps1(attrib2Vec3.x);
            __m128 sseAttrib1Z = _mm_set_ps1(attrib2Vec3.y);
            __m128 sseAttrib2Z = _mm_set_ps1(attrib2Vec3.z);

            __m128 sseVec4AttribZ = _mm_add_ps(
                _mm_mul_ps(sseAttrib0Z, ssef0XY),
                _mm_add_ps(_mm_mul_ps(sseAttrib1Z, ssef1XY), sseAttrib2Z));

            // vec4::w attribute to be interpolated
            __m128 sseAttrib0W = _mm_set_ps1(attrib2Vec3.x);
            __m128 sseAttrib1W = _mm_set_ps1(attrib2Vec3.y);
            __m128 sseAttrib2W = _mm_set_ps1(attrib2Vec3.z);

            __m128 sseVec3AttribW = _mm_add_ps(
                _mm_mul_ps(sseAttrib0W, ssef0XY),
                _mm_add_ps(_mm_mul_ps(sseAttrib1W, ssef1XY), sseAttrib2W));

            pInterpolatedAttributes[i].m_Vec4Attributes->m_SSEX = sseVec4AttribX;
            pInterpolatedAttributes[i].m_Vec4Attributes->m_SSEY = sseVec4AttribY;
            pInterpolatedAttributes[i].m_Vec4Attributes->m_SSEZ = sseVec4AttribZ;
            pInterpolatedAttributes[i].m_Vec4Attributes->m_SSEW = sseVec3AttribW;
        }

        // vec3 xyz attributes
        for (uint32_t i = 0; i < m_pRenderEngine->m_ShaderMetadata.m_NumVec3Attributes; i++)
        {
            // Fetch interpolation deltas computed after VS was returned
            const glm::vec3& attrib0Vec3 = m_pRenderEngine->m_SetupBuffers.m_Attribute3Deltas[i][(primIdx * 3) + 0];
            const glm::vec3& attrib1Vec3 = m_pRenderEngine->m_SetupBuffers.m_Attribute3Deltas[i][(primIdx * 3) + 1];
            const glm::vec3& attrib2Vec3 = m_pRenderEngine->m_SetupBuffers.m_Attribute3Deltas[i][(primIdx * 3) + 2];

            // vec3::x attribute to be interpolated
            __m128 sseAttrib0X = _mm_set_ps1(attrib0Vec3.x);
            __m128 sseAttrib1X = _mm_set_ps1(attrib0Vec3.y);
            __m128 sseAttrib2X = _mm_set_ps1(attrib0Vec3.z);

            __m128 sseVec3AttribX = _mm_add_ps(
                _mm_mul_ps(sseAttrib0X, ssef0XY),
                _mm_add_ps(_mm_mul_ps(sseAttrib1X, ssef1XY), sseAttrib2X));

            // vec3::y attribute to be interpolated
            __m128 sseAttrib0Y = _mm_set_ps1(attrib1Vec3.x);
            __m128 sseAttrib1Y = _mm_set_ps1(attrib1Vec3.y);
            __m128 sseAttrib2Y = _mm_set_ps1(attrib1Vec3.z);

            __m128 sseVec3AttribY = _mm_add_ps(
                _mm_mul_ps(sseAttrib0Y, ssef0XY),
                _mm_add_ps(_mm_mul_ps(sseAttrib1Y, ssef1XY), sseAttrib2Y));

            // vec3::z attribute to be interpolated
            __m128 sseAttrib0Z = _mm_set_ps1(attrib2Vec3.x);
            __m128 sseAttrib1Z = _mm_set_ps1(attrib2Vec3.y);
            __m128 sseAttrib2Z = _mm_set_ps1(attrib2Vec3.z);

            __m128 sseVec3AttribZ = _mm_add_ps(
                _mm_mul_ps(sseAttrib0Z, ssef0XY),
                _mm_add_ps(_mm_mul_ps(sseAttrib1Z, ssef1XY), sseAttrib2Z));

            pInterpolatedAttributes[i].m_Vec3Attributes->m_SSEX = sseVec3AttribX;
            pInterpolatedAttributes[i].m_Vec3Attributes->m_SSEY = sseVec3AttribY;
            pInterpolatedAttributes[i].m_Vec3Attributes->m_SSEZ = sseVec3AttribZ;
        }

        // vec2 xy attributes
        for (uint32_t i = 0; i < m_pRenderEngine->m_ShaderMetadata.m_NumVec2Attributes; i++)
        {
            // Fetch interpolation deltas computed after VS was returned
            const glm::vec3& attrib0Vec3 = m_pRenderEngine->m_SetupBuffers.m_Attribute2Deltas[i][(primIdx * 2) + 0];
            const glm::vec3& attrib1Vec3 = m_pRenderEngine->m_SetupBuffers.m_Attribute2Deltas[i][(primIdx * 2) + 1];
            const glm::vec3& attrib2Vec3 = m_pRenderEngine->m_SetupBuffers.m_Attribute2Deltas[i][(primIdx * 2) + 2];

            // vec3::x attribute to be interpolated
            __m128 sseAttrib0X = _mm_set_ps1(attrib0Vec3.x);
            __m128 sseAttrib1X = _mm_set_ps1(attrib0Vec3.y);
            __m128 sseAttrib2X = _mm_set_ps1(attrib0Vec3.z);

            __m128 sseVec2AttribX = _mm_add_ps(
                _mm_mul_ps(sseAttrib0X, ssef0XY),
                _mm_add_ps(_mm_mul_ps(sseAttrib1X, ssef1XY), sseAttrib2X));

            // vec3::y attribute to be interpolated
            __m128 sseAttrib0Y = _mm_set_ps1(attrib1Vec3.x);
            __m128 sseAttrib1Y = _mm_set_ps1(attrib1Vec3.y);
            __m128 sseAttrib2Y = _mm_set_ps1(attrib1Vec3.z);

            __m128 sseVec2AttribY = _mm_add_ps(
                _mm_mul_ps(sseAttrib0Y, ssef0XY),
                _mm_add_ps(_mm_mul_ps(sseAttrib1Y, ssef1XY), sseAttrib2Y));

            pInterpolatedAttributes[i].m_Vec2Attributes->m_SSEX = sseVec2AttribX;
            pInterpolatedAttributes[i].m_Vec2Attributes->m_SSEY = sseVec2AttribY;
        }
    }
}