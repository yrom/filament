/*
 * Copyright (C) 2022 The Android Open Source Project
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

#ifndef TNT_FILAMENT_BACKEND_VULKANPIPELINECACHE_H
#define TNT_FILAMENT_BACKEND_VULKANPIPELINECACHE_H

#include <backend/DriverEnums.h>
#include <backend/TargetBufferInfo.h>

#include "backend/Program.h"

#include <bluevk/BlueVK.h>

#include <utils/bitset.h>
#include <utils/compiler.h>
#include <utils/Hash.h>

#include <list>
#include <tsl/robin_map.h>
#include <type_traits>
#include <vector>
#include <unordered_map>

#include "VulkanCommands.h"

VK_DEFINE_HANDLE(VmaAllocator)
VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaPool)

namespace filament::backend {

struct VulkanProgram;
struct VulkanBufferObject;
struct VulkanTexture;
class VulkanResourceAllocator;

// VulkanPipelineCache manages a cache of descriptor sets and pipelines.
//
// Please note the following limitations:
//
// - Push constants are not supported. (if adding support, see VkPipelineLayoutCreateInfo)
// - Only DESCRIPTOR_TYPE_COUNT descriptor sets are bound at a time.
// - Assumes that viewport and scissor should be dynamic. (not baked into VkPipeline)
// - Assumes that uniform buffers should be visible across all shader stages.
//
class VulkanPipelineCache : public CommandBufferObserver {
public:
    VulkanPipelineCache(VulkanPipelineCache const&) = delete;
    VulkanPipelineCache& operator=(VulkanPipelineCache const&) = delete;

    static constexpr uint32_t UBUFFER_BINDING_COUNT = Program::UNIFORM_BINDING_COUNT;
    static constexpr uint32_t SAMPLER_BINDING_COUNT = MAX_SAMPLER_COUNT;

    // We assume only one possible input attachment between two subpasses. See also the subpasses
    // definition in VulkanFboCache.
    static constexpr uint32_t INPUT_ATTACHMENT_COUNT = 1;

    static constexpr uint32_t SHADER_MODULE_COUNT = 2;
    static constexpr uint32_t VERTEX_ATTRIBUTE_COUNT = MAX_VERTEX_ATTRIBUTE_COUNT;

    // Three descriptor set layouts: uniforms, combined image samplers, and input attachments.
    static constexpr uint32_t DESCRIPTOR_TYPE_COUNT = 3;
    static constexpr uint32_t INITIAL_DESCRIPTOR_SET_POOL_SIZE = 512;

    using UsageFlags = utils::bitset128;
    static UsageFlags getUsageFlags(uint16_t binding, ShaderStageFlags stages, UsageFlags src = {});
    static UsageFlags disableUsageFlags(uint16_t binding, UsageFlags src);

    #pragma clang diagnostic push
    #pragma clang diagnostic warning "-Wpadded"

    // The RasterState POD contains standard graphics-related state like blending, culling, etc.
    // The following states are omitted because Filament never changes them:
    // >>> depthClampEnable, rasterizerDiscardEnable, depthBoundsTestEnable, stencilTestEnable
    // >>> minSampleShading, alphaToOneEnable, sampleShadingEnable, minDepthBounds, maxDepthBounds,
    // >>> depthBiasClamp, polygonMode, lineWidth
    struct RasterState {
        VkCullModeFlags       cullMode : 2;
        VkFrontFace           frontFace : 2;
        VkBool32              depthBiasEnable : 1;
        VkBool32              blendEnable : 1;
        VkBool32              depthWriteEnable : 1;
        VkBool32              alphaToCoverageEnable : 1;
        VkBlendFactor         srcColorBlendFactor : 5; // offset = 1 byte
        VkBlendFactor         dstColorBlendFactor : 5;
        VkBlendFactor         srcAlphaBlendFactor : 5;
        VkBlendFactor         dstAlphaBlendFactor : 5;
        VkColorComponentFlags colorWriteMask : 4;
        uint8_t               rasterizationSamples;    // offset = 4 bytes
        uint8_t               colorTargetCount;        // offset = 5 bytes
        BlendEquation         colorBlendOp : 4;        // offset = 6 bytes
        BlendEquation         alphaBlendOp : 4;
        SamplerCompareFunc    depthCompareOp;          // offset = 7 bytes
        float                 depthBiasConstantFactor; // offset = 8 bytes
        float                 depthBiasSlopeFactor;    // offset = 12 bytes
    };

    static_assert(std::is_trivially_copyable<RasterState>::value,
            "RasterState must be a POD for fast hashing.");

    static_assert(sizeof(RasterState) == 16, "RasterState must not have implicit padding.");

    struct UniformBufferBinding {
        VkBuffer buffer;
        VkDeviceSize offset;
        VkDeviceSize size;
    };

    // Upon construction, the pipeCache initializes some internal state but does not make any Vulkan
    // calls. On destruction it will free any cached Vulkan objects that haven't already been freed.
    VulkanPipelineCache(VulkanResourceAllocator* allocator);
    ~VulkanPipelineCache();
    void setDevice(VkDevice device, VmaAllocator allocator);

    // Creates new descriptor sets if necessary and binds them using vkCmdBindDescriptorSets.
    // Returns false if descriptor set allocation fails.
    bool bindDescriptors(VkCommandBuffer cmdbuffer) noexcept;

    // Creates a new pipeline if necessary and binds it using vkCmdBindPipeline.
    // Returns false if an error occurred.
    bool bindPipeline(VulkanCommandBuffer* commands) noexcept;

    // Sets up a new scissor rectangle if it has been dirtied.
    void bindScissor(VkCommandBuffer cmdbuffer, VkRect2D scissor) noexcept;

    // Each of the following methods are fast and do not make Vulkan calls.
    void bindProgram(VulkanProgram* program) noexcept;
    void bindRasterState(const RasterState& rasterState) noexcept;
    void bindRenderPass(VkRenderPass renderPass, int subpassIndex) noexcept;
    void bindPrimitiveTopology(VkPrimitiveTopology topology) noexcept;
    void bindUniformBufferObject(uint32_t bindingIndex, VulkanBufferObject* bufferObject,
            VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) noexcept;
    void bindUniformBuffer(uint32_t bindingIndex, VkBuffer buffer,
            VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) noexcept;
    void bindSamplers(VkDescriptorImageInfo samplers[SAMPLER_BINDING_COUNT],
            VulkanTexture* textures[SAMPLER_BINDING_COUNT], UsageFlags flags) noexcept;
    void bindInputAttachment(uint32_t bindingIndex, VkDescriptorImageInfo imageInfo) noexcept;
    void bindVertexArray(VkVertexInputAttributeDescription const* attribDesc,
            VkVertexInputBindingDescription const* bufferDesc, uint8_t count);

    // Gets the current UBO at the given slot, useful for push / pop.
    UniformBufferBinding getUniformBufferBinding(uint32_t bindingIndex) const noexcept;

    // Checks if the given uniform is bound to any slot, and if so binds "null" to that slot.
    // Also invalidates all cached descriptors that refer to the given buffer.
    // This is only necessary when the client knows that the UBO is about to be destroyed.
    void unbindUniformBuffer(VkBuffer uniformBuffer) noexcept;

    // Checks if an image view is bound to any sampler, and if so resets that particular slot.
    // Also invalidates all cached descriptors that refer to the given image view.
    // This is only necessary when the client knows that a texture is about to be destroyed.
    void unbindImageView(VkImageView imageView) noexcept;

    // NOTE: In theory we should proffer "unbindSampler" but in practice we never destroy samplers.

    // Destroys all managed Vulkan objects. This should be called before changing the VkDevice.
    void terminate() noexcept;

    // vkCmdBindPipeline and vkCmdBindDescriptorSets establish bindings to a specific command
    // buffer; they are not global to the device. Therefore we need to be notified when a
    // new command buffer becomes active.
    void onCommandBuffer(const VulkanCommandBuffer& cmdbuffer) override;

    // Injects a dummy texture that can be used to clear out old descriptor sets.
    void setDummyTexture(VkImageView imageView) {
        mDummyTargetInfo.imageView = imageView;
    }

    // Acquires a resource to be bound to the current pipeline. The ownership of the resource
    // will be transferred to the corresponding pipeline when pipeline is bound.
    void acquireResource(VulkanResource* resource) {
        mPipelineBoundResources.acquire(resource);
    }

    inline RasterState getCurrentRasterState() const noexcept {
        return mCurrentRasterState;
    }

    // We need to update this outside of bindRasterState due to VulkanDriver::draw.
    inline void setCurrentRasterState(RasterState const& rasterState) noexcept {
        mCurrentRasterState = rasterState;
    }

private:
    // PIPELINE LAYOUT CACHE KEY
    // -------------------------

    using PipelineLayoutKey = utils::bitset128;

    static_assert(PipelineLayoutKey::BIT_COUNT >= 2 * MAX_SAMPLER_COUNT);

    struct PipelineLayoutKeyHashFn {
        size_t operator()(const PipelineLayoutKey& key) const;
    };

    struct PipelineLayoutKeyEqual {
        bool operator()(const PipelineLayoutKey& k1, const PipelineLayoutKey& k2) const;
    };

    // PIPELINE CACHE KEY
    // ------------------

    // Equivalent to VkVertexInputAttributeDescription but half as big.
    struct VertexInputAttributeDescription {
        VertexInputAttributeDescription& operator=(const VkVertexInputAttributeDescription& that) {
            assert_invariant(that.location <= 0xffu);
            assert_invariant(that.binding <= 0xffu);
            assert_invariant(uint32_t(that.format) <= 0xffffu);
            location = that.location;
            binding = that.binding;
            format = that.format;
            offset = that.offset;
            return *this;
        }
        operator VkVertexInputAttributeDescription() const {
            return { location, binding, VkFormat(format), offset };
        }
        uint8_t     location;
        uint8_t     binding;
        uint16_t    format;
        uint32_t    offset;
    };

    // Equivalent to VkVertexInputBindingDescription but not as big.
    struct VertexInputBindingDescription {
        VertexInputBindingDescription& operator=(const VkVertexInputBindingDescription& that) {
            assert_invariant(that.binding <= 0xffffu);
            binding = that.binding;
            stride = that.stride;
            inputRate = that.inputRate;
            return *this;
        }
        operator VkVertexInputBindingDescription() const {
            return { binding, stride, (VkVertexInputRate) inputRate };
        }
        uint16_t binding;
        uint16_t inputRate;
        uint32_t stride;
    };

    // The pipeline key is a POD that represents all currently bound states that form the immutable
    // VkPipeline object. The size:offset comments below are expressed in bytes.
    struct PipelineKey {                                                          // size : offset
        VkShaderModule shaders[SHADER_MODULE_COUNT];                              //  16  : 0
        VkRenderPass renderPass;                                                  //  8   : 16
        uint16_t topology;                                                        //  2   : 24
        uint16_t subpassIndex;                                                    //  2   : 26
        VertexInputAttributeDescription vertexAttributes[VERTEX_ATTRIBUTE_COUNT]; //  128 : 28
        VertexInputBindingDescription vertexBuffers[VERTEX_ATTRIBUTE_COUNT];      //  128 : 156
        RasterState rasterState;                                                  //  16  : 284
        uint32_t padding;                                                         //  4   : 300
        PipelineLayoutKey layout;                                                 // 16   : 304
    };

    static_assert(sizeof(PipelineKey) == 320, "PipelineKey must not have implicit padding.");

    using PipelineHashFn = utils::hash::MurmurHashFn<PipelineKey>;

    struct PipelineEqual {
        bool operator()(const PipelineKey& k1, const PipelineKey& k2) const;
    };

    // CACHE ENTRY STRUCTS
    // -------------------

    // The timestamp associated with a given cache entry represents time as a count of flush
    // events since the cache was constructed. If any cache entry was most recently used over
    // FVK_MAX_PIPELINE_AGE flushes in the past, then we can be sure that it is no longer
    // being used by the GPU, and is therefore safe to destroy or reclaim.
    using Timestamp = uint64_t;
    Timestamp mCurrentTime = 0;

    struct PipelineCacheEntry {
        VkPipeline handle;
        Timestamp lastUsed;
    };

    struct PipelineLayoutCacheEntry {
        VkPipelineLayout handle;
        Timestamp lastUsed;

        std::array<VkDescriptorSetLayout, DESCRIPTOR_TYPE_COUNT> descriptorSetLayouts;

        // Each pipeline layout has 3 arenas of unused descriptors (one for each binding type).
        //
        // The difference between the "arenas" and the "pool" are as follows.
        //
        // - The "pool" is a single, centralized factory for all descriptors (VkDescriptorPool).
        //
        // - Each "arena" is a set of unused (but alive) descriptors that can only be used with a
        //   specific pipeline layout and a specific binding type. We manually manage each arena.
        //   The arenas are created in an empty state, and they are gradually populated as new
        //   descriptors are reclaimed over time.  This is quite different from the pool, which is
        //   given a fixed size when it is constructed.
        //
        std::array<std::vector<VkDescriptorSet>, DESCRIPTOR_TYPE_COUNT> descriptorSetArenas;
    };

    // CACHE CONTAINERS
    // ----------------

    using PipelineLayoutMap = tsl::robin_map<PipelineLayoutKey , PipelineLayoutCacheEntry,
            PipelineLayoutKeyHashFn, PipelineLayoutKeyEqual>;
    using PipelineMap = tsl::robin_map<PipelineKey, PipelineCacheEntry,
            PipelineHashFn, PipelineEqual>;

    PipelineLayoutMap mPipelineLayouts;
    PipelineMap mPipelines;

    PipelineCacheEntry* createPipeline() noexcept;
    PipelineLayoutCacheEntry* getOrCreatePipelineLayout() noexcept;

    // Immutable state.
    VkDevice mDevice = VK_NULL_HANDLE;
    VmaAllocator mAllocator = VK_NULL_HANDLE;

    // Current requirements for the pipeline layout, pipeline, and descriptor sets.
    RasterState mCurrentRasterState;
    PipelineKey mPipelineRequirements = {};
    VkSpecializationInfo const* mSpecializationRequirements = nullptr;

    // Current bindings for the pipeline and descriptor sets.
    PipelineKey mBoundPipeline = {};

    // Current state for scissoring.
    VkRect2D mCurrentScissor = {};
    
    VkBuffer mDummyBuffer;
    VmaAllocation mDummyMemory;

    VulkanResourceAllocator* mResourceAllocator;
    VulkanAcquireOnlyResourceManager mPipelineBoundResources;
};

} // namespace filament::backend

#endif // TNT_FILAMENT_BACKEND_VULKANPIPELINECACHE_H
