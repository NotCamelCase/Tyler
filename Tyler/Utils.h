#pragma once

namespace tyler
{
    // General utilities

//#define LOG_ENABLED

#ifdef _DEBUG
#define ASSERT(x) assert(x)
#else
#define ASSERT(x)
#endif

#ifdef LOG_ENABLED
#define LOG(...) do { printf(__VA_ARGS__); } while(false)
#else
#define LOG(...) 
#endif

#define EDGE_TEST_SHARED_EDGES

    struct Rect2D
    {
        float   m_MinX;
        float   m_MinY;
        float   m_MaxX;
        float   m_MaxY;
    };

    // Scalar debug function to evaluate edge function E(x, y) and sample(x, y) with tie-breaking rules
    static bool EvaluateEdgeFunction(const glm::vec3& E, const glm::vec2& sample)
    {
        // Interpolate edge function at given sample
        float result = (sample.x * E.x) + (sample.y * E.y) + E.z;
#ifdef EDGE_TEST_SHARED_EDGES
        // Apply tie-breaking rules on shared vertices in order to avoid double-shading fragments
        if (result > 0.f) return true;
        else if (result < 0.f) return false;

        if (E.x > 0.f) return true;
        else if (E.x < 0.f) return false;

        if ((E.x == 0.0f) && (E.y < 0.f)) return false;
        else return true;
#else
        return result >= 0.f;
#endif
    }
}