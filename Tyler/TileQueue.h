#pragma once

namespace tyler
{
    static constexpr uint32_t   g_scInvalidTileIndex = 0xffffffff;

    // A tile is a rectangular subregion of a frame buffer
    struct Tile
    {
        Tile() {}
        Tile(const Tile& other) { m_IsTileQueued.clear(std::memory_order_relaxed); }
        Tile& operator=(const Tile& other)
        {
            m_PosX = other.m_PosX;
            m_PosY = other.m_PosY;

            m_IsTileQueued.clear(std::memory_order_relaxed);

            return *this;
        }

        // Position of tile within framebuffer
        float               m_PosX;
        float               m_PosY;

        // Indicates if the tile is already queued for rasterization,
        // which is done once when a tile receives its first input primitive
        std::atomic_flag    m_IsTileQueued;
    };

    // Atomically-operated fixed-size FIFO of tile indices which
    // will be used to rasterize binned primitives to tiles
    struct TileQueue
    {
        ~TileQueue()
        {
            delete[] m_pData;
        }

        // Allocate fixed-size FIFO memory for total number of tiles; must be re-allocated when tile count changes!
        void AllocateBackingMemory(uint32_t totalTileCount)
        {
            // We don't need the stale data in m_pData in case it needs to be resized,
            // so we simply delete it and re-allocate manually for current totalTileCount
            delete[] m_pData;

            m_DataSize = totalTileCount;

            // Allocate backing memory
            m_pData = new uint32_t[m_DataSize];

            ResetQueue();
        }

        // Reset it for next drawcall
        void ResetQueue()
        {
            // Reset running indices
            m_ReadIdx.store(0u, std::memory_order_relaxed);
            m_WriteIdx.store(0u, std::memory_order_relaxed);
            m_FetchIdx.store(0u, std::memory_order_relaxed);

            // Clear tile indices assigned previously
            memset(m_pData, g_scInvalidTileIndex, sizeof(uint32_t) * m_DataSize);
        }

        // Store tileIdx and increment writeIdx
        void InsertTileIndex(uint32_t tileIdx)
        {
            ASSERT(tileIdx < m_DataSize);

            uint32_t prevTail = m_WriteIdx.fetch_add(1, std::memory_order_seq_cst);
            ASSERT(prevTail < m_DataSize);

            m_pData[prevTail] = tileIdx;
        }

        // Return next tileIdx and decrement readIdx
        uint32_t RemoveTileIndex()
        {
            uint32_t prevHead = m_ReadIdx.fetch_add(1, std::memory_order_seq_cst);
            ASSERT(prevHead < m_DataSize);

            return m_pData[prevHead];
        }

        // Return next tileIdx and decrement fetchIdx
        uint32_t FetchNextTileIndex()
        {
            uint32_t prevReadIdx = m_FetchIdx.fetch_add(1, std::memory_order_seq_cst);
            ASSERT(prevReadIdx < m_DataSize);

            return m_pData[prevReadIdx];
        }

        // Running indices of current read/write/fetch elements
        std::atomic<uint32_t>   m_ReadIdx;
        std::atomic<uint32_t>   m_WriteIdx;
        std::atomic<uint32_t>   m_FetchIdx;

        // Backing memory for tile indices
        uint32_t*               m_pData = nullptr;
        uint32_t                m_DataSize = 0u;
    };
}