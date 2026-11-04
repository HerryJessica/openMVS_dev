/*
* FuseDepthMapsCUDA.h
*
* GPU (CUDA) implementation of DepthMapsData::FuseDepthMaps.
*
* Host-visible declarations only: POD structs + plain C++ launcher functions.
* All __global__ / __device__ code lives in FuseDepthMapsCUDA.cu.
*/

#ifndef _MVS_FUSEDEPTHMAPSCUDA_H_
#define _MVS_FUSEDEPTHMAPSCUDA_H_

#include "Common.h"
#include "DepthMap.h"

#ifdef _USE_CUDA

	#include "FuseDepthMapsCUDA.inl" // POD structs + launcher decls (cuda-safe)

namespace MVS {

// ---------------------------------------------------------------------------
// Host-side manager. Owns the device cache + LRU, upload/download, dispatch.
// ---------------------------------------------------------------------------
class FuseDepthMapsCUDA
{
	public:
	FuseDepthMapsCUDA();
	~FuseDepthMapsCUDA();

	bool Init();
	void Release();

	// Upload one depth-map (with optional normal + confidence) to GPU memory.
	// Returns true on success; caches by idxImage (idempotent).
	bool UploadDepthData(IIndex idxImage,
	                     const DepthMap& depthMap,
	                     const NormalMap& normalMap,
	                     const ConfidenceMap& confMap,
	                     const Camera& camera,
	                     bool bNeedNormal);

	void SetPinnedImages(const IIndexArr& idxImages);
	bool IsCached(IIndex idxImage) const;
	// Reserve the complete per-image workspace before pinning depth-maps.
	bool PrepareFusionBuffers(size_t nPixels, size_t nNeighbors);

	// Fuse one reference depth-map into `pointcloud`.
	bool FuseOne(IIndex idxImage,
	             const IIndexArr& neighbors,
	             unsigned nMinViewsFuse,
	             float depthThresh,
	             float normalCosThresh,
	             bool bEstimateColor,
	             bool bEstimateNormal,
	             const ImageArr& images,
	             PointCloud& pointcloud);

	void SetMaxMemory(size_t maxBytes);
	size_t GetUsedMemory() const;

	private:
	struct Impl;
	Impl* impl_;
};

} // namespace MVS

#endif // _USE_CUDA
#endif // _MVS_FUSEDEPTHMAPSCUDA_H_