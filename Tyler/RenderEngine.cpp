#include "RenderEngine.h"

#include "PipelineThread.h"

namespace tyler
{
    RenderEngine::RenderEngine(const RasterizerConfig& renderConfig)
        :
        m_RenderConfig(renderConfig)
    {
        //TODO: Init everything properly!

        // Allocate triangle setup data big enough to hold all possible in-flight primitives
        m_SetupBuffers.m_pEdgeCoefficients = new glm::vec3[m_RenderConfig.m_MaxDrawIterationSize * 3 /* 3 vertices */];
        m_SetupBuffers.m_pInterpolatedZValues = new glm::vec3[m_RenderConfig.m_MaxDrawIterationSize];

        // Allocate memory for bounding boxed to be cached after Binning
        m_SetupBuffers.m_pPrimBBoxes = new Rect2D[m_RenderConfig.m_MaxDrawIterationSize];

        // Allocate memory for interpolation related data
        for (uint32_t i = 0; i < g_scMaxVertexAttributes; i++)
        {
            m_SetupBuffers.m_Attribute4Deltas[i] = new glm::vec3[m_RenderConfig.m_MaxDrawIterationSize * 4 /*xyzw*/];
            m_SetupBuffers.m_Attribute3Deltas[i] = new glm::vec3[m_RenderConfig.m_MaxDrawIterationSize * 3 /*xyz*/];
            m_SetupBuffers.m_Attribute2Deltas[i] = new glm::vec3[m_RenderConfig.m_MaxDrawIterationSize * 2 /*xy*/];
        }

        //TODO: Attributes interpolation deltas!

        // Create PipelineThreads that will spawn their own worker thread to implement the pipeline stages in parallel
        m_PipelineThreads.resize(m_RenderConfig.m_NumPipelineThreads);
        for (uint32_t idx = 0; idx < m_RenderConfig.m_NumPipelineThreads; idx++)
        {
            PipelineThread* pThread = new PipelineThread(this, idx);
            m_PipelineThreads[idx] = pThread;
        }
    }

    RenderEngine::~RenderEngine()
    {
        // Clean up dynamic allocations

        for (PipelineThread* pThread : m_PipelineThreads)
        {
            delete pThread;
        }

        for (auto& perTileCoverageMask : m_CoverageMasks)
        {
            for (auto& perThreadCoverageMasks : perTileCoverageMask)
            {
                delete perThreadCoverageMasks;
            }
        }

        // Triangle setup buffers
        delete[] m_SetupBuffers.m_pEdgeCoefficients;
        delete[] m_SetupBuffers.m_pPrimBBoxes;

        for (uint32_t i = 0; i < g_scMaxVertexAttributes; i++)
        {
            delete[] m_SetupBuffers.m_Attribute4Deltas[i];
            delete[] m_SetupBuffers.m_Attribute3Deltas[i];
            delete[] m_SetupBuffers.m_Attribute2Deltas[i];
        }

        delete[] m_SetupBuffers.m_pInterpolatedZValues;
    }

    void RenderEngine::ClearRenderTargets(bool clearColor, const glm::vec4& colorValue, bool clearDepth, float depthValue)
    {
        ASSERT(!clearColor || (m_Framebuffer.m_pColorBuffer != nullptr));
        ASSERT(!clearDepth || (m_Framebuffer.m_pDepthBuffer != nullptr));

        //TODO: Clear frame buffer tiles!!!
        //TODO: Optimize render target clearing!!!

        const glm::uvec4 color = {
            static_cast<uint8_t>(colorValue.x * 255.f),
            static_cast<uint8_t>(colorValue.y * 255.f),
            static_cast<uint8_t>(colorValue.z * 255.f),
            static_cast<uint8_t>(colorValue.w * 255.f),
        };

        if (clearColor)
        {
            const uint32_t fbSize = m_Framebuffer.m_Width * m_Framebuffer.m_Height * 4 /*R8G8B8A8_UNORM*/;

            for (uint32_t i = 0; i < fbSize; i += 4)
            {
                // Surface format == R8G8B8A8_UNORM

                m_Framebuffer.m_pColorBuffer[i + 0] = color.x;
                m_Framebuffer.m_pColorBuffer[i + 1] = color.y;
                m_Framebuffer.m_pColorBuffer[i + 2] = color.z;
                m_Framebuffer.m_pColorBuffer[i + 3] = color.w;
            }
        }

        if (clearDepth)
        {
            const uint32_t fbSize = m_Framebuffer.m_Width * m_Framebuffer.m_Height * 1 /*D32_FLOAT*/;

            for (uint32_t i = 0; i < fbSize; i++)
            {
                m_Framebuffer.m_pDepthBuffer[i] = depthValue;
            }
        }
    }

    void RenderEngine::SetRenderTargets(Framebuffer* pFramebuffer)
    {
        //TODO: Check active framebuffer for any meaningful change (e.g. RT resolution) before allocating RT-dependent data!
        //TODO: NULL RT for color/depth for color-only or depth-only rendering?!

        if ((pFramebuffer->m_Width != m_Framebuffer.m_Width) ||
            (pFramebuffer->m_Height != m_Framebuffer.m_Height))
        {
            // Set active render area and RTs
            m_Framebuffer.m_Width = pFramebuffer->m_Width;
            m_Framebuffer.m_Height = pFramebuffer->m_Height;
            m_Framebuffer.m_pColorBuffer = pFramebuffer->m_pColorBuffer;
            m_Framebuffer.m_pDepthBuffer = pFramebuffer->m_pDepthBuffer;

            ASSERT((m_Framebuffer.m_Width > 0u) && (m_Framebuffer.m_Height > 0u));

            //TODO: Only resize vector when necessary!!!
            //TODO: Initialize frame buffer tiles with given width and height!!!

            // Determine total number of tiles to be allocated
            uint32_t numTileX = static_cast<uint32_t>(glm::ceil(static_cast<float>(m_Framebuffer.m_Width) / m_RenderConfig.m_TileSize));
            uint32_t numTileY = static_cast<uint32_t>(glm::ceil(static_cast<float>(m_Framebuffer.m_Height) / m_RenderConfig.m_TileSize));

            m_NumTilePerRow = numTileX;
            m_NumTilePerColumn = numTileY;

            uint32_t totalTileCount = numTileX * numTileY;

            // Resize the array for total number of tiles needed for current RT
            m_TileList.resize(totalTileCount);

            // Initialize tiles
            for (uint32_t y = 0; y < numTileY; y++)
            {
                for (uint32_t x = 0; x < numTileX; x++)
                {
                    Tile tile;
                    tile.m_PosX = static_cast<float>(glm::min(m_Framebuffer.m_Width, x * m_RenderConfig.m_TileSize));
                    tile.m_PosY = static_cast<float>(glm::min(m_Framebuffer.m_Height, y * m_RenderConfig.m_TileSize));
                    // Tile::m_IsTileQueued flag will be cleared pre-draw iteration

                    m_TileList[GetGlobalTileIndex(x, y)] = tile;
                }
            }

            //TODO: Only resize vector when necessary!!!

            // Configure array of bins based on RT and tile size
            // m_BinList[TILE_COUNT][THREAD_COUNT][PRIM_COUNT]
            m_BinList.resize(totalTileCount);
            for (uint32_t i = 0; i < totalTileCount; i++)
            {
                m_BinList[i].resize(m_RenderConfig.m_NumPipelineThreads);

                for (uint32_t j = 0; j < m_RenderConfig.m_NumPipelineThreads; j++)
                {
                    // Only reserve the memory for per-thread primitive indices!
                    m_BinList[i][j].reserve(m_RenderConfig.m_MaxDrawIterationSize / m_RenderConfig.m_NumPipelineThreads);
                }
            }

            // Configure array of coverage masks buffer
            //m_CoverageMasks[TILE_COUNT][THREAD_COUNT]
            m_CoverageMasks.resize(totalTileCount);
            for (uint32_t i = 0; i < totalTileCount; i++)
            {
                m_CoverageMasks[i].resize(m_RenderConfig.m_NumPipelineThreads);

                for (uint32_t j = 0; j < m_RenderConfig.m_NumPipelineThreads; j++)
                {
                    CoverageMaskBuffer* pBuffer = new CoverageMaskBuffer(m_RenderConfig.m_TileSize);
                    m_CoverageMasks[i][j] = pBuffer;
                }
            }

            // Allocate rasterizer queue
            m_RasterizerQueue.AllocateBackingMemory(totalTileCount);
        }
    }

    void RenderEngine::DrawIndexed(uint32_t indexCount, uint32_t vertexOffset)
    {
        // Prepare for next drawcall
        ApplyPreDrawcallStateInvalidations();

        // Pipeline threads must have been allocated!
        ASSERT(m_PipelineThreads.size() == m_RenderConfig.m_NumPipelineThreads);

        // Only primitive topology type == TRIANGLE
        ASSERT((indexCount % 3) == 0);

        uint32_t numTotalPrims = indexCount / 3;
        uint32_t numRemainingPrims = numTotalPrims;

        uint32_t drawElemsPrev = 0u;
        uint32_t numIter = 0;

        while (numRemainingPrims > 0)
        {
            // Prepare for next draw iteration
            ApplyPreDrawIterationStateInvalidations();

            //TODO: What if (numTotalPrims < numThreads)!!!

            // How many prims are to be processed this iteration & prims per thread
            uint32_t iterationSize = (numRemainingPrims >= m_RenderConfig.m_MaxDrawIterationSize) ? m_RenderConfig.m_MaxDrawIterationSize : numRemainingPrims;
            uint32_t perIterationRemainder = iterationSize % m_RenderConfig.m_NumPipelineThreads;
            uint32_t primsPerThread = iterationSize / m_RenderConfig.m_NumPipelineThreads;

            for (uint32_t threadIdx = 0; threadIdx < m_RenderConfig.m_NumPipelineThreads; threadIdx++)
            {
                uint32_t currentDrawElemsStart = drawElemsPrev;
                uint32_t currentDrawElemsEnd = (threadIdx == (m_RenderConfig.m_NumPipelineThreads - 1)) ?
                    // If number of remaining primitives in iteration is not multiple of number of threads, have the last thread cover the remaining range
                    (currentDrawElemsStart + primsPerThread + perIterationRemainder) :
                    currentDrawElemsStart + primsPerThread;

                ASSERT(currentDrawElemsEnd <= numTotalPrims);

                // Threads must have been initialized and idle by now!
                PipelineThread* pThread = m_PipelineThreads[threadIdx];
                ASSERT((pThread != nullptr) && (pThread->m_CurrentState.load() == ThreadStatus::IDLE));

                // Assign computed draw elems range for thread
                pThread->m_ActiveDrawParams.m_ElemsStart = currentDrawElemsStart;
                pThread->m_ActiveDrawParams.m_ElemsEnd = currentDrawElemsEnd;
                pThread->m_ActiveDrawParams.m_VertexOffset = vertexOffset;

                LOG("Thread %d drawparams for iteration %d: (%d, %d)\n", threadIdx, numIter, currentDrawElemsStart, currentDrawElemsEnd);

                // PipelineThread drawcall input prepared, it can start processing of drawcall
                pThread->m_CurrentState.store(ThreadStatus::DRAWCALL_TOP, std::memory_order_release);

                drawElemsPrev = currentDrawElemsEnd;
                numRemainingPrims -= (currentDrawElemsEnd - currentDrawElemsStart);
            }

            // Stall main thread until all active threads complete given draw iteration
            WaitForPipelineThreadsToCompleteProcessingDrawcall();

            LOG("Iteration %d completed!\n", numIter++);
        }

#if _DEBUG
        // All threads must be idle and ready for next drawcall at this point
        for (PipelineThread* pThread : m_PipelineThreads)
        {
            ASSERT(pThread->m_CurrentState.load() == ThreadStatus::IDLE);
        }
#endif
    }

    void RenderEngine::ApplyPreDrawcallStateInvalidations()
    {
        //TODO

        // Clear VS$ data of each thread
        for (PipelineThread* pThread : m_PipelineThreads)
        {
            pThread->m_NumVertexCacheEntries = 0u;

#ifdef _DEBUG
            memset(pThread->m_CachedVertexIndices, UINT32_MAX, sizeof(uint32_t) * g_scVertexShaderCacheSize);
            memset(pThread->m_VertexCacheEntries, 0x0, sizeof(PipelineThread::VertexCache) * g_scVertexShaderCacheSize);

            memset(pThread->m_TempVertexAttributes, 0x0, 3 * sizeof(VertexAttributes));
#endif
        }
    }

    void RenderEngine::ApplyPreDrawIterationStateInvalidations()
    {
        ASSERT((m_NumTilePerColumn > 0) && (m_NumTilePerRow > 0));
        ASSERT(!m_TileList.empty());

        // Clear tile flags in allocated tiles list
        for (Tile& tile : m_TileList)
        {
            tile.m_IsTileQueued.clear(std::memory_order_relaxed);
        }

        ASSERT(!m_BinList.empty());

        // Clear binned primitives list
        for (auto& perTileList : m_BinList)
        {
            ASSERT(!perTileList.empty());

            for (auto& perThreadList : perTileList)
            {
                perThreadList.clear();
            }
        }

        // Reset coverage mask buffers
        for (auto& perThreadCoverageMask : m_CoverageMasks)
        {
            for (auto& coverageMaskBuffer : perThreadCoverageMask)
            {
                coverageMaskBuffer->ResetAllocationList();
            }
        }

        // Reset rasterizer queue
        m_RasterizerQueue.ResetQueue();
    }

    void RenderEngine::WaitForPipelineThreadsToCompleteProcessingDrawcall() const
    {
        bool drawcallComplete = false;
        while (!drawcallComplete) // Spin until active threads reach the bottom of drawcall pipeline
        {
            bool threadComplete = true;
            for (uint32_t i = 0; i < m_RenderConfig.m_NumPipelineThreads; i++)
            {
                PipelineThread* pThread = m_PipelineThreads[i];
                ASSERT(pThread != nullptr);

                threadComplete = threadComplete &&
                    (pThread->m_CurrentState.load(std::memory_order_relaxed) == ThreadStatus::DRAWCALL_BOTTOM);
            }

            drawcallComplete = threadComplete;
        }

        // All threads finished drawcall processing, set their states to IDLE before next iteration
        for (uint32_t i = 0; i < m_RenderConfig.m_NumPipelineThreads; i++)
        {
            PipelineThread* pThread = m_PipelineThreads[i];
            ASSERT((pThread != nullptr) && (pThread->m_CurrentState.load() == ThreadStatus::DRAWCALL_BOTTOM));

            pThread->m_CurrentState.store(ThreadStatus::IDLE, std::memory_order_relaxed);
        }
    }

    void RenderEngine::WaitForPipelineThreadsToCompleteBinning() const
    {
        bool binningComplete = false;
        while (!binningComplete) // Spin until active threads reach post binner
        {
            bool threadsComplete = true;
            for (uint32_t i = 0; i < m_RenderConfig.m_NumPipelineThreads; i++)
            {
                PipelineThread* pThread = m_PipelineThreads[i];
                ASSERT(pThread != nullptr);

                ThreadStatus expected = ThreadStatus::DRAWCALL_SYNC_POINT_POST_BINNER;
                threadsComplete = threadsComplete &&
                    (pThread->m_CurrentState.compare_exchange_weak(expected, ThreadStatus::DRAWCALL_RASTERIZATION, std::memory_order_acq_rel) || (expected >= ThreadStatus::DRAWCALL_RASTERIZATION));

                ASSERT((expected >= ThreadStatus::DRAWCALL_TOP) && (expected <= ThreadStatus::DRAWCALL_FRAGMENTSHADER));
            }

            binningComplete = threadsComplete;
        }
    }

    void RenderEngine::WaitForPipelineThreadsToCompleteRasterization() const
    {
        bool rasterComplete = false;
        while (!rasterComplete) // Spin until active threads reach post raster
        {
            bool threadsComplete = true;
            for (uint32_t i = 0; i < m_RenderConfig.m_NumPipelineThreads; i++)
            {
                PipelineThread* pThread = m_PipelineThreads[i];
                ASSERT(pThread->m_CurrentState.load() != ThreadStatus::IDLE);

                ThreadStatus expected = ThreadStatus::DRAWCALL_SYNC_POINT_POST_RASTER;
                threadsComplete = threadsComplete &&
                    (pThread->m_CurrentState.compare_exchange_weak(expected, ThreadStatus::DRAWCALL_FRAGMENTSHADER, std::memory_order_acq_rel) || (expected >= ThreadStatus::DRAWCALL_FRAGMENTSHADER));

                ASSERT((expected >= ThreadStatus::DRAWCALL_RASTERIZATION) && (expected <= ThreadStatus::DRAWCALL_BOTTOM));
            }

            rasterComplete = threadsComplete;
        }
    }

    void RenderEngine::EnqueueTileForRasterization(uint32_t tileIdx)
    {
        // Append the tile to the rasterizer queue if not already done
        if (!m_TileList[tileIdx].m_IsTileQueued.test_and_set(std::memory_order_acq_rel))
        {
            // Tile not queued up for rasterization, do so now
            m_RasterizerQueue.InsertTileIndex(tileIdx);
        }
    }

    void RenderEngine::BinPrimitiveForTile(uint32_t threadIdx, uint32_t tileIdx, uint32_t primIdx)
    {
        // Add primIdx to the per-thread bin of a tile

        std::vector<uint32_t>& tileBin = m_BinList[tileIdx][threadIdx];

        if (tileBin.empty())
        {
            // First encounter of primitive for tile, enqueue it for rasterization
            EnqueueTileForRasterization(tileIdx);
        }
        else
        {
            // Tile must have been already appended to the work queue
            ASSERT(m_TileList[tileIdx].m_IsTileQueued.test_and_set());
        }

        auto capPreAppend = tileBin.capacity();

        // Append primIdx to the tile's bin
        tileBin.push_back(primIdx);

        auto capPostAppend = tileBin.capacity();

        // Underlying tile bin vector must not have caused resizing when appending primitives to tiles!
        ASSERT(capPreAppend == capPostAppend);
    }

    void RenderEngine::AppendCoverageMask(uint32_t threadIdx, uint32_t tileIdx, const CoverageMask& mask)
    {
        m_CoverageMasks[tileIdx][threadIdx]->AppendCoverageMask(mask);
    }

    void RenderEngine::ResizeCoverageMaskBuffer(uint32_t threadIdx, uint32_t tileIdx)
    {
        m_CoverageMasks[tileIdx][threadIdx]->IncreaseCapacityIfNeeded();
    }
}