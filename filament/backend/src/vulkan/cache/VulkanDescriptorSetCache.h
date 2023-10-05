/*
 * Copyright (C) 2023 The Android Open Source Project
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

#ifndef TNT_FILAMENT_BACKEND_VULKANDESCRIPTORSETCACHE_H
#define TNT_FILAMENT_BACKEND_VULKANDESCRIPTORSETCACHE_H

namespace filament::backend {

class VulkanDescriptorSetCache : public CommandBufferObserver {
 public:

 private:

    // The descriptor set cache entry is a group of descriptor sets that are bound simultaneously.
    struct DescriptorCacheEntry {
        std::array<VkDescriptorSet, DESCRIPTOR_TYPE_COUNT> handles;
        Timestamp lastUsed;
        PipelineLayoutKey pipelineLayout;
        uint32_t id;
    };
    uint32_t mDescriptorCacheEntryCount = 0;

    // These helpers all return unstable pointers that should not be stored.
    DescriptorCacheEntry* createDescriptorSets() noexcept;

    // Misc helper methods.
    void destroyLayoutsAndDescriptors() noexcept;
    VkDescriptorPool createDescriptorPool(uint32_t size) const;
    void growDescriptorPool() noexcept;
    
    // DESCRIPTOR SET CACHE KEY
    // ------------------------

    // Equivalent to VkDescriptorImageInfo but with explicit padding.
    struct DescriptorImageInfo {
        DescriptorImageInfo& operator=(const VkDescriptorImageInfo& that) {
            sampler = that.sampler;
            imageView = that.imageView;
            imageLayout = that.imageLayout;
            padding = 0;
            return *this;
        }
        operator VkDescriptorImageInfo() const { return { sampler, imageView, imageLayout }; }

        // TODO: replace the 64-bit sampler handle with `uint32_t samplerParams` and remove the
        // padding field. This is possible if we have access to the VulkanSamplerCache.
        VkSampler sampler;

        VkImageView imageView;
        VkImageLayout imageLayout;
        uint32_t padding;
    };

    #pragma clang diagnostic pop

    
    // Represents all the Vulkan state that comprises a bound descriptor set.
    struct DescriptorKey {
        VkBuffer uniformBuffers[UBUFFER_BINDING_COUNT];               //   80     0
        DescriptorImageInfo samplers[SAMPLER_BINDING_COUNT];          // 1488    80
        DescriptorImageInfo inputAttachments[INPUT_ATTACHMENT_COUNT]; //   24  1568
        uint32_t uniformBufferOffsets[UBUFFER_BINDING_COUNT];         //   40  1592
        uint32_t uniformBufferSizes[UBUFFER_BINDING_COUNT];           //   40  1632
    };
    static_assert(offsetof(DescriptorKey, samplers)              == 80);
    static_assert(offsetof(DescriptorKey, inputAttachments)      == 1568);
    static_assert(offsetof(DescriptorKey, uniformBufferOffsets)  == 1592);
    static_assert(offsetof(DescriptorKey, uniformBufferSizes)    == 1632);
    static_assert(sizeof(DescriptorKey) == 1672, "DescriptorKey must not have implicit padding.");

    using DescHashFn = utils::hash::MurmurHashFn<DescriptorKey>;

    struct DescEqual {
        bool operator()(const DescriptorKey& k1, const DescriptorKey& k2) const;
    };

    using DescriptorMap
            = tsl::robin_map<DescriptorKey, DescriptorCacheEntry, DescHashFn, DescEqual>;
    using DescriptorResourceMap
            = std::unordered_map<uint32_t, std::unique_ptr<VulkanAcquireOnlyResourceManager>>;

    DescriptorMap mDescriptorSets;
    DescriptorResourceMap mDescriptorResources;

    // This describes the number of descriptor sets in mDescriptorPool. Note that this needs to be
    // multiplied by DESCRIPTOR_TYPE_COUNT to get the actual number of descriptor sets. Also note
    // that the number of low-level "descriptors" (not descriptor *sets*) is actually much more than
    // this size. It can be computed only by factoring in UBUFFER_BINDING_COUNT etc.
    uint32_t mDescriptorPoolSize = INITIAL_DESCRIPTOR_SET_POOL_SIZE;

    // To get the actual number of descriptor sets that have been allocated from the pool,
    // take the sum of mDescriptorArenasCount (these are inactive descriptor sets) and the
    // number of entries in the mDescriptorPool map (active descriptor sets). Multiply the result by
    // DESCRIPTOR_TYPE_COUNT.
    uint32_t mDescriptorArenasCount = 0;

    // The descriptor set pool starts out with a decent number of descriptor sets.  The cache can
    // grow the pool by re-creating it with a larger size.  See growDescriptorPool().
    VkDescriptorPool mDescriptorPool;

    // After a growth event (i.e. when the VkDescriptorPool is replaced with a bigger version), all
    // currently used descriptors are moved into the "extinct" sets so that they can be safely
    // destroyed a few frames later.
    std::list<VkDescriptorPool> mExtinctDescriptorPools;
    std::list<DescriptorCacheEntry> mExtinctDescriptorBundles;

    VkDescriptorBufferInfo mDummyBufferInfo = {};
    VkWriteDescriptorSet mDummyBufferWriteInfo = {};
    VkDescriptorImageInfo mDummyTargetInfo = {};
    VkWriteDescriptorSet mDummyTargetWriteInfo = {};

    DescriptorKey mDescriptorRequirements = {};    
    DescriptorKey mBoundDescriptor = {};    

    VulkanResourceAllocator* mResourceAllocator;    
};

} // filament::backend

#endif // TNT_FILAMENT_BACKEND_VULKANDESCRIPTORSETCACHE_H
