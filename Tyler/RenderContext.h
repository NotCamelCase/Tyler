#pragma once

#include "RasterizerConfig.h"
#include "RenderState.h"

namespace tyler
{
    struct RenderEngine;

    // Provides a rudimentary graphics API for underlying rasterizer
    class RenderContext
    {
    public:
        RenderContext(const RasterizerConfig& config);
        ~RenderContext();

        // Initialize RenderEngine
        bool Initialize();

        // Bind active color/depth buffers to be used in subsequent render pass
        void BindFramebuffer(Framebuffer* pFramebuffer);

        // Clear render targets, if requested
        void BeginRenderPass(bool clearColor, const glm::vec4& colorValue, bool clearDepth, float depthValue);

        // Set active vertex buffer and input stride for next drawcall
        void BindVertexBuffer(VertexBuffer* pVertexBuffer, uint32_t stride);

        // Set active index buffer for next drawcall
        void BindIndexBuffer(IndexBuffer* pIndexBuffer);

        // Bind pointer to constant buffer to be passed to VS/FS
        void BindConstantBuffer(ConstantBuffer* pConstantBuffer);

        // Bind shaders and shaders metada to be used
        void BindShaders(VertexShader vertexShader, FragmentShader fragmentShader, const ShaderMetadata& metadata);

        // Drawcalls
        void DrawIndexed(uint32_t indexCount, uint32_t vertexOffset);
        void Draw(uint32_t vertexCount) { /*TODO*/ }

        void EndRenderPass();

        // Shutdown @RenderEngine/subsystems and free all dynamically alloc'd memory
        void Destroy();

    private:
        RenderEngine*       m_pRenderEngine = nullptr;
        RasterizerConfig    m_Config;
    };
}