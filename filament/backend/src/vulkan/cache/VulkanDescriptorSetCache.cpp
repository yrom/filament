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

#include "VuklanDescriptorSetCache.h"

VuklanDescriptorSetCache::DescriptorCacheEntry* VuklanDescriptorSetCache::createDescriptorSets() noexcept {
    PipelineLayoutCacheEntry* layoutCacheEntry = getOrCreatePipelineLayout();

    DescriptorCacheEntry descriptorCacheEntry = {
        .pipelineLayout = mPipelineRequirements.layout,
        .id = mDescriptorCacheEntryCount++,
    };

    // Each of the arenas for this particular layout are guaranteed to have the same size. Check
    // the first arena to see if any descriptor sets are available that can be re-claimed. If not,
    // create brand new ones (one for each type). They will be added to the arena later, after they
    // are no longer used. This occurs during the cleanup phase during command buffer submission.
    auto& descriptorSetArenas = layoutCacheEntry->descriptorSetArenas;
    if (descriptorSetArenas[0].empty()) {

        // If allocating a new descriptor set from the pool would cause it to overflow, then
        // recreate the pool. The number of descriptor sets that have already been allocated from
        // the pool is the sum of the "active" descriptor sets (mDescriptorSets) and the "dormant"
        // descriptor sets (mDescriptorArenasCount).
        //
        // NOTE: technically both sides of the inequality below should be multiplied by
        // DESCRIPTOR_TYPE_COUNT to get the true number of descriptor sets.
        if (mDescriptorSets.size() + mDescriptorArenasCount + 1 > mDescriptorPoolSize) {
            growDescriptorPool();
        }

        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = mDescriptorPool;
        allocInfo.descriptorSetCount = DESCRIPTOR_TYPE_COUNT;
        allocInfo.pSetLayouts = layoutCacheEntry->descriptorSetLayouts.data();
        VkResult error = vkAllocateDescriptorSets(mDevice, &allocInfo,
                descriptorCacheEntry.handles.data());
        assert_invariant(error == VK_SUCCESS);
        if (error != VK_SUCCESS) {
            return nullptr;
        }
    } else {
        for (uint32_t i = 0; i < DESCRIPTOR_TYPE_COUNT; ++i) {
            descriptorCacheEntry.handles[i] = descriptorSetArenas[i].back();
            descriptorSetArenas[i].pop_back();
        }
        assert_invariant(mDescriptorArenasCount > 0);
        mDescriptorArenasCount--;
    }

    // Rewrite every binding in the new descriptor sets.
    VkDescriptorBufferInfo descriptorBuffers[UBUFFER_BINDING_COUNT];
    VkDescriptorImageInfo descriptorSamplers[SAMPLER_BINDING_COUNT];
    VkDescriptorImageInfo descriptorInputAttachments[INPUT_ATTACHMENT_COUNT];
    VkWriteDescriptorSet descriptorWrites[UBUFFER_BINDING_COUNT + SAMPLER_BINDING_COUNT +
            INPUT_ATTACHMENT_COUNT];
    uint32_t nwrites = 0;
    VkWriteDescriptorSet* writes = descriptorWrites;
    nwrites = 0;
    for (uint32_t binding = 0; binding < UBUFFER_BINDING_COUNT; binding++) {
        VkWriteDescriptorSet& writeInfo = writes[nwrites++];
        if (mDescriptorRequirements.uniformBuffers[binding]) {
            VkDescriptorBufferInfo& bufferInfo = descriptorBuffers[binding];
            bufferInfo.buffer = mDescriptorRequirements.uniformBuffers[binding];
            bufferInfo.offset = mDescriptorRequirements.uniformBufferOffsets[binding];
            bufferInfo.range = mDescriptorRequirements.uniformBufferSizes[binding];

            // We store size with 32 bits, so our "WHOLE" sentinel is different from Vk.
            if (bufferInfo.range == WHOLE_SIZE) {
                bufferInfo.range = VK_WHOLE_SIZE;
            }

            writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeInfo.pNext = nullptr;
            writeInfo.dstArrayElement = 0;
            writeInfo.descriptorCount = 1;
            writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeInfo.pImageInfo = nullptr;
            writeInfo.pBufferInfo = &bufferInfo;
            writeInfo.pTexelBufferView = nullptr;
        } else {
            writeInfo = mDummyBufferWriteInfo;
            assert_invariant(mDummyBufferWriteInfo.pBufferInfo->buffer);
        }
        assert_invariant(writeInfo.pBufferInfo->buffer);
        writeInfo.dstSet = descriptorCacheEntry.handles[0];
        writeInfo.dstBinding = binding;
    }
    for (uint32_t binding = 0; binding < SAMPLER_BINDING_COUNT; binding++) {
        if (mDescriptorRequirements.samplers[binding].sampler) {
            VkWriteDescriptorSet& writeInfo = writes[nwrites++];
            VkDescriptorImageInfo& imageInfo = descriptorSamplers[binding];
            imageInfo = mDescriptorRequirements.samplers[binding];
            writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeInfo.pNext = nullptr;
            writeInfo.dstArrayElement = 0;
            writeInfo.descriptorCount = 1;
            writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeInfo.pImageInfo = &imageInfo;
            writeInfo.pBufferInfo = nullptr;
            writeInfo.pTexelBufferView = nullptr;
            writeInfo.dstSet = descriptorCacheEntry.handles[1];
            writeInfo.dstBinding = binding;
        }
    }
    for (uint32_t binding = 0; binding < INPUT_ATTACHMENT_COUNT; binding++) {
        if (mDescriptorRequirements.inputAttachments[binding].imageView) {
            VkWriteDescriptorSet& writeInfo = writes[nwrites++];
            VkDescriptorImageInfo& imageInfo = descriptorInputAttachments[binding];
            imageInfo = mDescriptorRequirements.inputAttachments[binding];
            writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeInfo.pNext = nullptr;
            writeInfo.dstArrayElement = 0;
            writeInfo.descriptorCount = 1;
            writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            writeInfo.pImageInfo = &imageInfo;
            writeInfo.pBufferInfo = nullptr;
            writeInfo.pTexelBufferView = nullptr;
            writeInfo.dstSet = descriptorCacheEntry.handles[2];
            writeInfo.dstBinding = binding;
        }
    }

    vkUpdateDescriptorSets(mDevice, nwrites, writes, 0, nullptr);

    return &mDescriptorSets.emplace(mDescriptorRequirements, descriptorCacheEntry).first.value();
}

bool VuklanDescriptorSetCache::bindDescriptors(VkCommandBuffer cmdbuffer) noexcept {
    DescriptorMap::iterator descriptorIter = mDescriptorSets.find(mDescriptorRequirements);

    // Check if the required descriptors are already bound. If so, there's no need to do anything.
    if (DescEqual equals; UTILS_LIKELY(equals(mBoundDescriptor, mDescriptorRequirements))) {

        // If the pipeline state during an app's first draw call happens to match the default state
        // vector of the cache, then the cache is uninitialized and we should not return early.
        if (UTILS_LIKELY(!mDescriptorSets.empty())) {

            // Since the descriptors are already bound, they should be found in the cache.
            assert_invariant(descriptorIter != mDescriptorSets.end());

            // Update the LRU "time stamp" (really a count of cmd buf submissions) before returning.
            descriptorIter.value().lastUsed = mCurrentTime;
            return true;
        }
    }

    // If a cached object exists, re-use it, otherwise create a new one.
    DescriptorCacheEntry* cacheEntry = UTILS_LIKELY(descriptorIter != mDescriptorSets.end()) ?
            &descriptorIter.value() : createDescriptorSets();

    // If a descriptor set overflow occurred, allow higher levels to handle it gracefully.
    assert_invariant(cacheEntry != nullptr);
    if (UTILS_UNLIKELY(cacheEntry == nullptr)) {
        return false;
    }

    cacheEntry->lastUsed = mCurrentTime;
    mBoundDescriptor = mDescriptorRequirements;
    // This passes the currently "bound" uniform buffer objects to pipeline that will be used in the
    // draw call.
    auto resourceEntry = mDescriptorResources.find(cacheEntry->id);
    if (resourceEntry == mDescriptorResources.end()) {
        mDescriptorResources[cacheEntry->id]
                = std::make_unique<VulkanAcquireOnlyResourceManager>(mResourceAllocator);
        resourceEntry = mDescriptorResources.find(cacheEntry->id);
    }
    resourceEntry->second->acquireAll(&mPipelineBoundResources);

    vkCmdBindDescriptorSets(cmdbuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            getOrCreatePipelineLayout()->handle, 0, VulkanPipelineCache::DESCRIPTOR_TYPE_COUNT,
            cacheEntry->handles.data(), 0, nullptr);

    return true;
}
