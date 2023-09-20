/*
 * Copyright (C) 2018 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TNT_FILAMENT_BACKEND_VULKANHANDLES_H
#define TNT_FILAMENT_BACKEND_VULKANHANDLES_H

// This needs to be at the top
#include "DriverBase.h"

#include "VulkanBuffer.h"
#include "VulkanPipelineCache.h"
#include "VulkanResources.h"
#include "VulkanSwapChain.h"
#include "VulkanTexture.h"
#include "VulkanUtility.h"

#include "private/backend/SamplerGroup.h"

#include <utils/Mutex.h>

namespace filament::backend {

class VulkanTimestamps;
class VulkanSamplerCache;

struct VulkanProgram : public HwProgram, VulkanResource {

    VulkanProgram(VkDevice device, const Program& builder) noexcept;
    VulkanProgram(VkDevice device, VkShaderModule vs, VkShaderModule fs,
            utils::FixedCapacityVector<std::tuple<uint8_t, uint8_t, ShaderStageFlags>> const&
                    samplerBindings) noexcept;
    ~VulkanProgram();

    inline VkShaderModule getVertexShader() const { return mInfo->shaders[0]; }

    inline VkShaderModule getFragmentShader() const { return mInfo->shaders[1]; }

    inline VulkanPipelineCache::UsageFlags getUsage() const { return mInfo->usage; }

    inline utils::FixedCapacityVector<uint16_t> const& getBindingToSamplerIndex() const {
        return mInfo->bindingToSamplerIndex;
    }

    inline VkSpecializationInfo const& getSpecConstInfo() const {
        return mInfo->specializationInfo;
    }

private:
    // TODO: handle compute shaders.
    // The expected order of shaders - from frontend to backend - is vertex, fragment, compute.
    static constexpr uint8_t MAX_SHADER_MODULES = 2;

    struct PipelineInfo {
        PipelineInfo(size_t specConstsCount) :
            bindingToSamplerIndex(MAX_SAMPLER_COUNT, 0xffff),
            specConsts(specConstsCount, VkSpecializationMapEntry{}),
            specConstData(new char[specConstsCount * 4])
        {}

        // This bitset maps to each of the sampler in the sampler groups associated with this
        // program, and whether each sampler is used in which shader (i.e. vert, frag, compute).
        VulkanPipelineCache::UsageFlags usage;

        // We store the samplerGroupIndex as the top 8-bit and the index within each group as the lower 8-bit.
        utils::FixedCapacityVector<uint16_t> bindingToSamplerIndex;
        VkShaderModule shaders[MAX_SHADER_MODULES] = {VK_NULL_HANDLE};
        VkSpecializationInfo specializationInfo = {};
        utils::FixedCapacityVector<VkSpecializationMapEntry> specConsts;
        std::unique_ptr<char[]> specConstData;
    };

    PipelineInfo* mInfo;
    VkDevice mDevice = VK_NULL_HANDLE;
};

// The render target bundles together a set of attachments, each of which can have one of the
// following ownership semantics:
//
// - The attachment's VkImage is shared and the owner is VulkanSwapChain (mOffscreen = false).
// - The attachment's VkImage is shared and the owner is VulkanTexture   (mOffscreen = true).
//
// We use private inheritance to shield clients from the width / height fields in HwRenderTarget,
// which are not representative when this is the default render target.
struct VulkanRenderTarget : private HwRenderTarget, VulkanResource {
    // Creates an offscreen render target.
    VulkanRenderTarget(VkDevice device, VkPhysicalDevice physicalDevice,
            VulkanContext const& context, VmaAllocator allocator,
            VulkanCommands* commands, uint32_t width, uint32_t height,
            uint8_t samples, VulkanAttachment color[MRT::MAX_SUPPORTED_RENDER_TARGET_COUNT],
            VulkanAttachment depthStencil[2], VulkanStagePool& stagePool);

    // Creates a special "default" render target (i.e. associated with the swap chain)
    explicit VulkanRenderTarget();

    void transformClientRectToPlatform(VkRect2D* bounds) const;
    void transformClientRectToPlatform(VkViewport* bounds) const;
    VkExtent2D getExtent() const;
    VulkanAttachment getColor(int target) const;
    VulkanAttachment getMsaaColor(int target) const;
    VulkanAttachment getDepth() const;
    VulkanAttachment getMsaaDepth() const;
    uint8_t getColorTargetCount(const VulkanRenderPass& pass) const;
    uint8_t getSamples() const { return mSamples; }
    bool hasDepth() const { return mDepth.texture; }
    bool isSwapChain() const { return !mOffscreen; }
    void bindToSwapChain(VulkanSwapChain& surf);

private:
    VulkanAttachment mColor[MRT::MAX_SUPPORTED_RENDER_TARGET_COUNT] = {};
    VulkanAttachment mDepth = {};
    VulkanAttachment mMsaaAttachments[MRT::MAX_SUPPORTED_RENDER_TARGET_COUNT] = {};
    VulkanAttachment mMsaaDepthAttachment = {};
    const bool mOffscreen : 1;
    uint8_t mSamples : 7;
};

struct VulkanBufferObject;

struct VulkanVertexBuffer : public HwVertexBuffer, VulkanResource {
    VulkanVertexBuffer(VulkanContext& context, VulkanStagePool& stagePool,
            VulkanResourceAllocator* allocator, uint8_t bufferCount, uint8_t attributeCount,
            uint32_t elementCount, AttributeArray const& attributes);

    void setBuffer(VulkanBufferObject* bufferObject, uint32_t index);

    utils::FixedCapacityVector<VulkanBuffer const*> buffers;

private:
    FixedSizeVulkanResourceManager<MAX_VERTEX_BUFFER_COUNT> mResources;
};

struct VulkanIndexBuffer : public HwIndexBuffer, VulkanResource {
    VulkanIndexBuffer(VmaAllocator allocator, VulkanStagePool& stagePool, uint8_t elementSize,
            uint32_t indexCount)
        : HwIndexBuffer(elementSize, indexCount),
          VulkanResource(VulkanResourceType::INDEX_BUFFER),
          buffer(allocator, stagePool, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, elementSize * indexCount),
          indexType(elementSize == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32) {}

    VulkanBuffer buffer;
    const VkIndexType indexType;
};

struct VulkanBufferObject : public HwBufferObject, VulkanResource {
    VulkanBufferObject(VmaAllocator allocator, VulkanStagePool& stagePool, uint32_t byteCount,
            BufferObjectBinding bindingType, BufferUsage usage);

    VulkanBuffer buffer;
    const BufferObjectBinding bindingType;
};

struct VulkanSamplerGroup : public HwSamplerGroup, VulkanResource {

    //using SamplerGroupResourceManager = FixedSizeVulkanResourceManager<MAX_SAMPLER_COUNT>;
    using SamplerGroupResourceManager = VulkanAcquireOnlyResourceManager;
    
    VulkanSamplerGroup(VulkanResourceAllocator* allocator, VulkanSamplerCache* samplerCache,
            size_t count);

    explicit VulkanSamplerGroup(
            utils::FixedCapacityVector<std::pair<VulkanTexture*, VkSampler>> const& samplers);

    ~VulkanSamplerGroup() {
        //        utils::slog.e <<"sample-group=" << this << " going out of scope" << utils::io::endl;
    }

    void update(SamplerDescriptor const* samplerGroup, size_t count);

    VkDescriptorImageInfo const* getSamplerInfo() const { return mInfo->imageInfo.data(); }

    inline bool hasDepthTexture(VulkanTexture* depthTexture) const {
        //        utils::slog.e <<"hasdepth texture=" << depthTexture << " vk=" << depthTexture->getVkImage() << utils::io::endl;
        //        for (auto d : mInfo->depthTextures) {
        //            utils::slog.e <<"d=" << d << " vk=" << d->getVkImage() << utils::io::endl;
        //        }
        auto ret = mInfo->depthTextures.find(depthTexture) != mInfo->depthTextures.end();
        //        utils::slog.e <<"hasdepth return=" << ret << utils::io::endl;
        return ret;
    }

    // Transition depth textures to samplers
    inline void transitionDepthSamplers(VkCommandBuffer cmdbuffer) const {
        for (auto texture: mInfo->depthTextures) {
            auto range = texture->getFullViewRange();
            // Transition each level one-by-one
            for (uint32_t i = range.baseMipLevel; i < range.levelCount; ++i) {
                texture->transitionLayout(cmdbuffer,
                                          {
                                              .aspectMask = range.aspectMask,
                                              .baseMipLevel = i,
                                              .levelCount = 1,
                                              .baseArrayLayer = range.baseArrayLayer,
                                              .layerCount = 1
                                          },
                                          VulkanLayout::DEPTH_SAMPLER);
            }
        }
    }

    // This is for copying the resources held by this sampler group into the descriptor set (since descriptor sets can be cached).
    SamplerGroupResourceManager& getResources() { return mResources; }
    VulkanTexture* const* getTextures() {
        return mInfo->textures.data();
    }

private:
    struct PipelineInfo {
        template<typename TYPE>
        using Array = utils::FixedCapacityVector<TYPE>;
        PipelineInfo(size_t size)
            : imageInfo(size, VkDescriptorImageInfo {}),
              textures(size, nullptr) {}
        Array<VkDescriptorImageInfo> imageInfo;
        Array<VulkanTexture*> textures;
        std::unordered_set<VulkanTexture*> depthTextures;
    };

    void update(size_t index);

    VulkanSamplerCache* mSamplerCache;
    VulkanResourceAllocator* mResourceAllocator;
    PipelineInfo* mInfo = nullptr;
    // MAX_SAMPLER_COUNT corresponds to the maximum number of textsures that the sampler group can
    // reference
    SamplerGroupResourceManager mResources;
};

struct VulkanRenderPrimitive : public HwRenderPrimitive, VulkanResource {
    VulkanRenderPrimitive(VulkanResourceAllocator* allocator)
        : VulkanResource(VulkanResourceType::RENDER_PRIMITIVE),
          mResources(allocator) {}

    ~VulkanRenderPrimitive() {
        mResources.clear();
    }

    void setPrimitiveType(PrimitiveType pt);
    void setBuffers(VulkanVertexBuffer* vertexBuffer, VulkanIndexBuffer* indexBuffer);
    VulkanVertexBuffer* vertexBuffer = nullptr;
    VulkanIndexBuffer* indexBuffer = nullptr;
    VkPrimitiveTopology primitiveTopology;

private:
    // Used to track the vertex and index buffers only.
    FixedSizeVulkanResourceManager<2> mResources;
};

struct VulkanFence : public HwFence, VulkanResource {
    VulkanFence()
        : VulkanResource(VulkanResourceType::FENCE) {}

    explicit VulkanFence(std::shared_ptr<VulkanCmdFence> fence)
        : VulkanResource(VulkanResourceType::FENCE),
          fence(fence) {}

    std::shared_ptr<VulkanCmdFence> fence;
};

struct VulkanTimerQuery : public HwTimerQuery, VulkanThreadSafeResource {
    explicit VulkanTimerQuery(std::tuple<uint32_t, uint32_t> indices);
    ~VulkanTimerQuery();

    void setFence(std::shared_ptr<VulkanCmdFence> fence) noexcept;

    bool isCompleted() noexcept;

    uint32_t getStartingQueryIndex() const {
        return mStartingQueryIndex;
    }

    uint32_t getStoppingQueryIndex() const {
        return mStoppingQueryIndex;
    }

private:
    uint32_t mStartingQueryIndex;
    uint32_t mStoppingQueryIndex;

    std::shared_ptr<VulkanCmdFence> mFence;
    utils::Mutex mFenceMutex;
};

inline constexpr VkBufferUsageFlagBits getBufferObjectUsage(
        BufferObjectBinding bindingType) noexcept {
    switch(bindingType) {
        case BufferObjectBinding::VERTEX:
            return VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        case BufferObjectBinding::UNIFORM:
            return VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        case BufferObjectBinding::SHADER_STORAGE:
            return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        // when adding more buffer-types here, make sure to update VulkanBuffer::loadFromCpu()
        // if necessary.
    }
}

} // namespace filament::backend

#endif // TNT_FILAMENT_BACKEND_VULKANHANDLES_H
