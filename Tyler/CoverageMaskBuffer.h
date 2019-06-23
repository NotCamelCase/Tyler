#pragma once

namespace tyler
{
    // Max number of times that a coverage mask buffer allocation will occur to increase available coverage masks
    static constexpr uint8_t    g_scMaxBufferSlots = 8u;

    static constexpr uint16_t   g_scQuadMask0 = (1 << 0);
    static constexpr uint16_t   g_scQuadMask1 = (1 << 1);
    static constexpr uint16_t   g_scQuadMask2 = (1 << 2);
    static constexpr uint16_t   g_scQuadMask3 = (1 << 3);

    enum class CoverageMaskType : uint16_t
    {
        TILE,
        BLOCK,
        QUAD
    };

    struct CoverageMask
    {
        // Sample positions to be fragment-shaded
        // LL corner position in case of tile/block masks
        // First sample's position in case of 4-fragment quad masks
        uint32_t            m_SampleX;
        uint32_t            m_SampleY;

        // Global primitive index for fetching EE coefficients, vertex attributes, depth data
        uint32_t            m_PrimIdx;

        // Type of coverage mask (TILE, BLOCK, QUAD)
        CoverageMaskType    m_Type;

        // 4-fragment coverage mask (only applies when m_Type is QUAD!)
        uint16_t            m_QuadMask;
    };

    // Per-tile masks buffers that the rasterizer will emit for primitives that need to be fragment-shaded
    struct CoverageMaskBuffer
    {
        CoverageMaskBuffer(uint32_t tileSize)
            :
            m_CurrentAllocationIdx(0),
            m_NumAllocations(0),
            m_SlotAllocationPendingSwap(false)
        {
            memset(m_AllocationList, 0x0, g_scMaxBufferSlots * sizeof(Slot));

            // Allocate and initalize first buffer slot
            Slot* pBuffer = &m_AllocationList[m_CurrentAllocationIdx];
            pBuffer->m_Capacity = tileSize * tileSize / 2; //TODO: Compute initial size based on tile size and RT resolution maybe?!
            pBuffer->m_pData = new CoverageMask[pBuffer->m_Capacity];

            ++m_NumAllocations;
        }

        ~CoverageMaskBuffer()
        {
            for (Slot& slot : m_AllocationList)
            {
                delete[] slot.m_pData;
            }
        }

        void AppendCoverageMask(const CoverageMask& mask)
        {
            // Get current buffer slot to which mask will be appended to
            Slot& currentSlot = m_AllocationList[m_CurrentAllocationIdx];
            ASSERT(currentSlot.m_pData != nullptr);

            if ((currentSlot.m_AllocationCount == currentSlot.m_Capacity)) // Slot full, switch to the next one
            {
                Slot& nextSlot = m_AllocationList[++m_CurrentAllocationIdx];
                ASSERT((nextSlot.m_pData != nullptr));
                ASSERT(nextSlot.m_AllocationCount == 0u);

                nextSlot.m_pData[nextSlot.m_AllocationCount++] = mask;

                // Next slot swapped in, next allocation can take place, if need be
                m_SlotAllocationPendingSwap = false;
            }
            else
            {
                currentSlot.m_pData[currentSlot.m_AllocationCount++] = mask;
            }
        }

        void IncreaseCapacityIfNeeded()
        {
            Slot& currentSlot = m_AllocationList[m_CurrentAllocationIdx];
            if (!m_SlotAllocationPendingSwap &&
                (currentSlot.m_AllocationCount >= (currentSlot.m_Capacity / 2)))
            {
                // Must not exceed max # of allocations
                ASSERT(m_NumAllocations < g_scMaxBufferSlots);

                // Current slot half-full (or half-empty :d), allocate memory for next slot ahead of time
                Slot& nextSlot = m_AllocationList[m_CurrentAllocationIdx + 1];
                if (nextSlot.m_Capacity == 0)
                {
                    ASSERT(nextSlot.m_pData == nullptr);

                    nextSlot.m_Capacity = currentSlot.m_Capacity * 2;
                    nextSlot.m_pData = new CoverageMask[nextSlot.m_Capacity]; // Double the size of current slot
                    ASSERT(nextSlot.m_AllocationCount == 0u);

                    m_SlotAllocationPendingSwap = true;

                    ++m_NumAllocations;
                }
                else
                {
                    ASSERT(nextSlot.m_pData != nullptr);

                    m_SlotAllocationPendingSwap = true;

                    // Buffer already allocated, just point to it
                    ++m_NumAllocations;
                }
            }
        }

        // Reset buffer allocations for next iteration
        void ResetAllocationList()
        {
            for (uint32_t i = 0; i < m_NumAllocations; i++)
            {
                Slot& slot = m_AllocationList[i];
                slot.m_AllocationCount = 0u;
            }

            // First slot is always allocated and used
            m_CurrentAllocationIdx = 0;
            m_NumAllocations = 1;

            m_SlotAllocationPendingSwap = false;
        }

        struct Slot
        {
            // Backing buffer memory
            CoverageMask*   m_pData;

            // Number of coverage masks the slot currently holds
            uint32_t        m_AllocationCount;

            // Total number of CoverageMasks available within the slot
            uint32_t        m_Capacity;
        };

        // Allocated slots of coverage masks buffer
        Slot        m_AllocationList[g_scMaxBufferSlots];
        // Index of current buffer slot to which new coverage masks will be appended
        uint32_t    m_CurrentAllocationIdx;
        // Number of coverage masks allocation occurred
        uint32_t    m_NumAllocations;
        // To prevent repeat allocations
        bool        m_SlotAllocationPendingSwap;
    };
}