#include "RenderContext.h"

#include "RasterizerConfig.h"
#include "RenderEngine.h"
#include "RenderState.h"

namespace tyler
{
    RenderContext::RenderContext(const RasterizerConfig& config) :
        m_Config(config)
    {
        if (m_Config.m_NumPipelineThreads == 0u)
        {
            // Set # of threads to HW threads, if param not provided
            m_Config.m_NumPipelineThreads = std::thread::hardware_concurrency() - 1;
        }
    }

    RenderContext::~RenderContext()
    {
    }

    bool RenderContext::Initialize()
    {
        m_pRenderEngine = new RenderEngine(m_Config);

        return true;
    }

    void RenderContext::Destroy()
    {
        delete m_pRenderEngine;
    }

    void RenderContext::BindFramebuffer(Framebuffer* pFrameBuffer)
    {
        ASSERT(pFrameBuffer != nullptr);

        // Have RenderEngine set up RT related data
        m_pRenderEngine->SetRenderTargets(pFrameBuffer);
    }

    void RenderContext::BeginRenderPass(bool clearColor, const glm::vec4& colorValue, bool clearDepth, float depthValue)
    {
        m_pRenderEngine->ClearRenderTargets(clearColor, colorValue, clearDepth, depthValue);
    }

    void RenderContext::BindVertexBuffer(VertexBuffer* pVertexBuffer, uint32_t stride)
    {
        ASSERT(pVertexBuffer != nullptr);
        ASSERT(stride > 0u);

        m_pRenderEngine->m_pVertexBuffer = pVertexBuffer;
        m_pRenderEngine->m_VertexInputStride = stride;
    }

    void RenderContext::BindIndexBuffer(IndexBuffer* pIndexBuffer)
    {
        ASSERT(pIndexBuffer != nullptr);
        m_pRenderEngine->m_pIndexBuffer = pIndexBuffer;
    }

    void RenderContext::BindConstantBuffer(ConstantBuffer* pConstantBuffer)
    {
        ASSERT(pConstantBuffer != nullptr);
        m_pRenderEngine->m_pConstantBuffer = pConstantBuffer;
    }

    void RenderContext::BindShaders(VertexShader vertexShader, FragmentShader fragmentShader, const ShaderMetadata& metadata)
    {
        // VS has to exist
        ASSERT(vertexShader != nullptr);

        m_pRenderEngine->m_VertexShader = vertexShader;
        m_pRenderEngine->m_FragmentShader = fragmentShader;
        m_pRenderEngine->m_ShaderMetadata = metadata;
    }

    void RenderContext::DrawIndexed(uint32_t indexCount, uint32_t vertexOffset)
    {
        //TODO: Non-indexed draw?!

        m_pRenderEngine->DrawIndexed(indexCount, vertexOffset);
    }

    void RenderContext::EndRenderPass()
    {
        //TODO
    }
}