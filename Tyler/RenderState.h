#pragma once

namespace tyler
{
    using IndexType = uint32_t;
    using IndexBuffer = IndexType;

    using VertexInput = void;
    using VertexBuffer = VertexInput;

    struct Framebuffer
    {
        // Better be 16-byte aligned for SSE, 32-byte for AVX!
        uint8_t*    m_pColorBuffer = nullptr; // Pre-defined color buffer format == R8G8B8A8_UNORM
        float*      m_pDepthBuffer = nullptr; // Pre-defined depth buffer format == D32_FLOAT

        uint32_t    m_Width = 0u;
        uint32_t    m_Height = 0u;
    };

    using ConstantBuffer = void;

    // Max number of vertex attributes that can be passed to FS after interpolation
    static constexpr uint8_t    g_scMaxVertexAttributes = 1u;

    struct VertexAttributes
    {
        glm::vec4   m_Attributes4[g_scMaxVertexAttributes];
        glm::vec3   m_Attributes3[g_scMaxVertexAttributes];
        glm::vec2   m_Attributes2[g_scMaxVertexAttributes];
    };

    // Packed group of attributes for 4 consecutive samples
    // that will be interpolated w/ SSE and passed onto FS
    struct InterpolatedAttributes
    {
        struct Vec4Attributes
        {
            union { glm::vec4 m_VecX; __m128 m_SSEX; };
            union { glm::vec4 m_VecY; __m128 m_SSEY; };
            union { glm::vec4 m_VecZ; __m128 m_SSEZ; };
            union { glm::vec4 m_VecW; __m128 m_SSEW; };
        };

        struct Vec3Attributes
        {
            union { glm::vec4 m_VecX; __m128 m_SSEX; };
            union { glm::vec4 m_VecY; __m128 m_SSEY; };
            union { glm::vec4 m_VecZ; __m128 m_SSEZ; };
        };

        struct Vec2Attributes
        {
            union { glm::vec4 m_VecX; __m128 m_SSEX; };
            union { glm::vec4 m_VecY; __m128 m_SSEY; };
        };

        Vec4Attributes  m_Vec4Attributes[g_scMaxVertexAttributes];
        Vec3Attributes  m_Vec3Attributes[g_scMaxVertexAttributes];
        Vec2Attributes  m_Vec2Attributes[g_scMaxVertexAttributes];
    };

    // VS/FS related shader metadata
    struct ShaderMetadata
    {
        // How many vec4/3/2 attributes are passed VS -> FS to be interpolated
        uint8_t    m_NumVec4Attributes;
        uint8_t    m_NumVec3Attributes;
        uint8_t    m_NumVec2Attributes;
    };

    // 4-sample fragment output
    struct FragmentOutput
    {
        // R32G32B32A32_FLOAT x 4
        __m128  m_FragmentColors[4];
    };

    // Vertex & Fragment shader definitions
    using VertexShader = glm::vec4(*)(VertexInput* pVertexInput, VertexAttributes* pVertexAttributes, ConstantBuffer* pConstantBuffer);
    using FragmentShader = void(*)(InterpolatedAttributes* pVertexAttributes, ConstantBuffer* pConstantBuffer, FragmentOutput* pFragmentOut);
}