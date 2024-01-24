/*
 * Copyright (C) 2019 The Android Open Source Project
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

#ifndef GLTFIO_FFILAMENTASSET_H
#define GLTFIO_FFILAMENTASSET_H

#include <gltfio/FilamentAsset.h>
#include <gltfio/NodeManager.h>
#include <gltfio/TrsTransformManager.h>

#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Texture.h>
#include <filament/TextureSampler.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>

#include <gltfio/MaterialProvider.h>
#include <gltfio/TextureProvider.h>

#include <math/mat4.h>

#include <utils/bitset.h>
#include <utils/FixedCapacityVector.h>
#include <utils/CString.h>
#include <utils/Entity.h>

#include <cgltf.h>

#include "downcast.h"
#include "DependencyGraph.h"
#include "DracoCache.h"
#include "FFilamentInstance.h"

#include <tsl/htrie_map.h>

#include <future>
#include <variant>
#include <vector>
#include <unordered_map>

#ifdef NDEBUG
#define GLTFIO_VERBOSE 0
#define GLTFIO_WARN(msg)
#else
#define GLTFIO_VERBOSE 1
#define GLTFIO_WARN(msg) slog.w << msg << io::endl
#endif

#if defined(__EMSCRIPTEN__) || defined(__ANDROID__) || defined(IOS)
#define GLTFIO_USE_FILESYSTEM 0
#else
#define GLTFIO_USE_FILESYSTEM 1
#endif

namespace utils {
    class NameComponentManager;
    class EntityManager;
}

namespace filament::gltfio {

struct Wireframe;

// This class is used to store the "temporary" buffers created before we have a VertexBuffer or
// MorphTargetBuffer. We cannot create VB or MTB beforehand because we do not know the number of
// vertices of the primitive until the tangents are computed (some methods will remesh the input).
template<typename BufferType>
struct BufferProducer {
    BufferProducer() {}

    // We delay the setting of count (either vertexCount or tangentCount) until the tangents are
    // computed
    void count(size_t count) { mCount = count; }

    void setExpectBufferAt(int slot) {
        assert_invariant(slot >= 0 && slot < 256);
        mExpected.set(static_cast<uint8_t>(slot));
    }

    std::future<BufferType> getFuture() { return std::move(mPromise.get_future()); }

protected:
    static_assert(MAX_MORPH_TARGETS <= 256);

    utils::bitset256 mExpected;
    std::promise<BufferType> mPromise;
    size_t mCount = 0;
};

struct VertexBufferProducer : public BufferProducer<VertexBuffer*> {

    // Override
    void setBufferObjectAt(FFilamentAsset* asset, Engine* engine, uint8_t slot,
            BufferObject* bufferObject);

    inline void setDummyObjectSlot(int dummyObjectSlot) { mDummyObjectSlot = dummyObjectSlot; }

private:
    std::unordered_map<uint8_t, BufferObject*> mBuffers;
    int mDummyObjectSlot = -1;
};

struct MorphTargetBufferProducer : public BufferProducer<MorphTargetBuffer*> {

    // Note that this will pass the ownership of the data arrays to this class.
    void setPositionAndTangentsAt(FFilamentAsset* asset, Engine* engine, int targetIndex,
            math::float3 const* positions, math::short4 const* tangents);

private:
    using PositionTangent = std::pair<math::float3 const*, math::short4 const*>;
    std::unordered_map<int, PositionTangent> mData;
};

struct IndexBufferProducer : public BufferProducer<IndexBuffer*> {

    // Note that the count() here will be the number of triangles.

    // Note that this will pass the ownership of the data arrays to this class.
    void setBuffer(FFilamentAsset* asset, Engine* engine, math::uint3 const* triangles);

    // Note that this will pass the ownership of the data arrays to this class.
    void setBuffer(FFilamentAsset* asset, Engine* engine, math::ushort3 const* triangles);
};

typedef std::shared_ptr<VertexBufferProducer> VertexBufferProducerPtr;
typedef std::shared_ptr<MorphTargetBufferProducer> MorphTargetBufferProducerPtr;
typedef std::shared_ptr<IndexBufferProducer> IndexBufferProducerPtr;

// Encapsulates the work to gather and upload the primitive buffers.
struct PrimitiveWorkload {
    PrimitiveWorkload(cgltf_primitive const* primitive) : primitive(primitive) {}

    cgltf_primitive const* primitive;

    IndexBufferProducerPtr indices;
    VertexBufferProducerPtr vertices;
    MorphTargetBufferProducerPtr targets;

    // The Filament Attribute is defined as a type, a slot, and whether the attribute is normalized or not.
    using FilamentAttribute = std::tuple<VertexAttribute, int, bool>;

    using Attribute = std::pair<cgltf_attribute_type, int>;
    std::unordered_map<Attribute, FilamentAttribute> attributesMap;
    bool generateFlatNormals = false;

    // A set of morph targets to generate tangents for.
    std::vector<int> morphTargets;

    static constexpr int const DUMMY_0 = -1;
    static constexpr int const DUMMY_1 = -2;
};

// Stores a connection between Texture and MaterialInstance; consumed by resource loader so that it
// can call "setParameter" on the given MaterialInstance after the Texture has been created.
// Since material instances are not typically shared between FilamentInstance, the slots are a
// unified list across all instances that exist before creation of Texture objects.
struct TextureSlot {
    MaterialInstance* materialInstance;
    const char* materialParameter;
};

// MeshCache
// ---------
// If a given glTF mesh is referenced by multiple glTF nodes, then it generates a separate Filament
// renderable for each of those nodes. All renderables generated by a given mesh share a common set
// of VertexBuffer and IndexBuffer objects. To achieve the sharing behavior, the loader maintains a
// small cache. There is one cache entry per cgltf_mesh. Each entry is a list of primitives, where a
// "primitive" is a reference to a Filament VertexBuffer and IndexBuffer.
struct Primitive {
    Aabb aabb; // object-space bounding box
    UvMap uvmap; // mapping from each glTF UV set to either UV0 or UV1 (8 bytes)

    std::future<MorphTargetBuffer*> targets;
    std::future<VertexBuffer*> vertices;
    std::future<IndexBuffer*> indices;

};
using MeshCache = utils::FixedCapacityVector<utils::FixedCapacityVector<Primitive>>;

struct FFilamentAsset : public FilamentAsset {
    FFilamentAsset(Engine* engine, utils::NameComponentManager* names,
            utils::EntityManager* entityManager, NodeManager* nodeManager,
            TrsTransformManager* trsTransformManager, const cgltf_data* srcAsset) :
            mEngine(engine), mNameManager(names), mEntityManager(entityManager),
            mNodeManager(nodeManager), mTrsTransformManager(trsTransformManager),
            mSourceAsset(new SourceAsset {(cgltf_data*)srcAsset}),
            mTextures(srcAsset->textures_count),
            mMeshCache(srcAsset->meshes_count) {}

    ~FFilamentAsset();

    size_t getEntityCount() const noexcept {
        return mEntities.size();
    }

    const utils::Entity* getEntities() const noexcept {
        return mEntities.empty() ? nullptr : mEntities.data();
    }

    const utils::Entity* getLightEntities() const noexcept {
        return mLightEntities.empty() ? nullptr : mLightEntities.data();
    }

    size_t getLightEntityCount() const noexcept {
        return mLightEntities.size();
    }

    const utils::Entity* getRenderableEntities() const noexcept {
        return (mRenderableCount == 0) ? nullptr : mEntities.data();
    }

    size_t getRenderableEntityCount() const noexcept {
        // Note that mRenderableCount is a "predicted" number of renderables, so if this is a
        // zero-instance asset, then we need to explicitly return zero.
        return mEntities.empty() ? 0 : mRenderableCount;
    }

    const utils::Entity* getCameraEntities() const noexcept {
        return mCameraEntities.empty() ? nullptr : mCameraEntities.data();
    }

    size_t getCameraEntityCount() const noexcept {
        return mCameraEntities.size();
    }

    utils::Entity getRoot() const noexcept {
        return mRoot;
    }

    size_t popRenderables(utils::Entity* entities, size_t count) noexcept {
        return mDependencyGraph.popRenderables(entities, count);
    }

    size_t getResourceUriCount() const noexcept {
        return mResourceUris.size();
    }

    const char* const* getResourceUris() const noexcept {
        return mResourceUris.data();
    }

    Aabb getBoundingBox() const noexcept {
        return mBoundingBox;
    }

    const char* getName(utils::Entity entity) const noexcept;

    const char* getExtras(utils::Entity entity) const noexcept;

    utils::Entity getFirstEntityByName(const char* name) noexcept;

    size_t getEntitiesByName(const char* name, utils::Entity* entities,
            size_t maxCount) const noexcept;

    size_t getEntitiesByPrefix(const char* prefix, utils::Entity* entities,
            size_t maxCount) const noexcept;

    const char* getMorphTargetNameAt(utils::Entity entity, size_t targetIndex) const noexcept;

    size_t getMorphTargetCountAt(utils::Entity entity) const noexcept;

    utils::Entity getWireframe() noexcept;

    Engine* getEngine() const noexcept {
        return mEngine;
    }

    TrsTransformManager* getTrsTransformManager() const noexcept {
        return mTrsTransformManager;
    }

    void releaseSourceData() noexcept;

    const void* getSourceAsset() const noexcept {
        return mSourceAsset.get() ? mSourceAsset->hierarchy : nullptr;
    }

    FilamentInstance** getAssetInstances() noexcept {
        return (FilamentInstance**) mInstances.data();
    }

    size_t getAssetInstanceCount() const noexcept {
        return mInstances.size();
    }

    size_t getSceneCount() const noexcept { return mScenes.size(); }

    const char* getSceneName(size_t sceneIndex) const noexcept {
        return mScenes[sceneIndex].c_str();
    }

    void addEntitiesToScene(Scene& targetScene, const Entity* entities, size_t count,
            SceneMask sceneFilter) const;

    void detachFilamentComponents() noexcept {
        mDetachedFilamentComponents = true;
    }

    // end public API

    // If a Filament Texture for the given args already exists, calls setParameter() and returns
    // early. If the Texture doesn't exist yet, stashes binding information for later.
    void addTextureBinding(MaterialInstance* materialInstance, const char* parameterName,
        const cgltf_texture* srcTexture, TextureProvider::TextureFlags flags);

    // Calls mi->setParameter() for the given texture slot and optionally adds an edge
    // to the dependency graph used for gradual reveal of entities.
    void applyTextureBinding(size_t textureIndex,const TextureSlot& tb, bool addDependency = true);

    struct Skin {
        utils::CString name;
        utils::FixedCapacityVector<math::mat4f> inverseBindMatrices;
    };

    Engine* const mEngine;
    utils::NameComponentManager* const mNameManager;
    utils::EntityManager* const mEntityManager;
    NodeManager* const mNodeManager;
    TrsTransformManager* const mTrsTransformManager;
    std::vector<utils::Entity> mEntities; // sorted such that renderables come first
    std::vector<utils::Entity> mLightEntities;
    std::vector<utils::Entity> mCameraEntities;
    size_t mRenderableCount = 0;
    std::vector<VertexBuffer*> mVertexBuffers;
    std::vector<BufferObject*> mBufferObjects;
    std::vector<IndexBuffer*> mIndexBuffers;
    std::vector<MorphTargetBuffer*> mMorphTargetBuffers;
    utils::FixedCapacityVector<Skin> mSkins;
    utils::FixedCapacityVector<utils::CString> mScenes;
    Aabb mBoundingBox;
    utils::Entity mRoot;
    std::vector<FFilamentInstance*> mInstances;
    Wireframe* mWireframe = nullptr;

    // Indicates if resource decoding has started (not necessarily finished)
    bool mResourcesLoaded = false;

    DependencyGraph mDependencyGraph;
    tsl::htrie_map<char, std::vector<utils::Entity>> mNameToEntity;
    utils::CString mAssetExtras;
    bool mDetachedFilamentComponents = false;

    // Encapsulates reference-counted source data, which includes the cgltf hierachy
    // and potentially also includes buffer data that can be uploaded to the GPU.
    struct SourceAsset {
        ~SourceAsset() { cgltf_free(hierarchy); }
        cgltf_data* hierarchy;
        DracoCache dracoCache;
        utils::FixedCapacityVector<uint8_t> glbData;
    };

    // We used shared ownership for the raw cgltf data in order to permit ResourceLoader to
    // complete various asynchronous work (e.g. uploading buffers to the GPU) even after the asset
    // or ResourceLoader have been destroyed.
    using SourceHandle = std::shared_ptr<SourceAsset>;
    SourceHandle mSourceAsset;

    // The mapping of root nodes to scene membership sets.
    tsl::robin_map<cgltf_node*, SceneMask> mRootNodes;

    // Stores all information related to a single cgltf_texture.
    // Note that more than one cgltf_texture can map to a single Filament texture,
    // e.g. if several have the same URL or bufferView. For each Filament texture,
    // only one of its corresponding TextureInfo slots will have isOwner=true.
    struct TextureInfo {
        std::vector<TextureSlot> bindings;
        Texture* texture;
        TextureProvider::TextureFlags flags;
        bool isOwner;
    };

    // Mapping from cgltf_texture to Texture* is required when creating new instances.
    utils::FixedCapacityVector<TextureInfo> mTextures;

    // Resource URIs can be queried by the end user.
    utils::FixedCapacityVector<const char*> mResourceUris;

    // The mapping from cgltf_mesh to VertexBuffer* (etc) is required when creating new instances.
    MeshCache mMeshCache;

    // Asset information that is produced by AssetLoader and consumed by ResourceLoader:
    std::vector<std::shared_ptr<PrimitiveWorkload>> mPrimitiveWorkloads;
};

FILAMENT_DOWNCAST(FilamentAsset)

} // namespace filament::gltfio

#endif // GLTFIO_FFILAMENTASSET_H
