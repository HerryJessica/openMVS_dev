/*
* FuseDepthMapsCUDA.cpp
*
* Host-side implementation of FuseDepthMapsCUDA. NO CUDA kernel syntax here;
* only host APIs (cudaMalloc/cudaMemcpy/...) and the launchers declared in the
* header. Compiled by the C++ compiler.
*/

#include "FuseDepthMapsCUDA.h"

#ifdef _USE_CUDA

	#include <cuda_runtime.h>
	#include <algorithm>
	#include <unordered_map>
	#include <unordered_set>
	#include <list>
	#include <climits>
	#include <limits>
	#include <mutex>
	#include <vector>

using namespace MVS;

// ------------------------- Impl -------------------------

struct FuseDepthMapsCUDA::Impl
{
	struct HostDepthState
	{
		std::vector<uint8_t> claimed;
		std::vector<uint8_t> invalid;
	};

	std::unordered_map<IIndex, GpuDepthData> cache;
	std::unordered_map<IIndex, HostDepthState> hostDepthState;
	std::unordered_set<SEACAVE::PairIdx> outputObservations;
	std::unordered_set<IIndex> pinnedImages;
	std::unordered_set<IIndex> touchedImages;
	std::list<IIndex> lru; // front = most recently used
	mutable std::mutex mtx;
	size_t maxMemory = 2ull * 1024 * 1024 * 1024; // 2 GB
	size_t usedMemory = 0;
	size_t temporaryMemory = 0;
	bool inited = false;

	FusionProposal* d_proposals = nullptr;
	ViewEntry* d_viewPool = nullptr;
	FusionProposal* d_out = nullptr;
	ViewEntry* d_outViewPool = nullptr;
	uint32_t* d_counts = nullptr;
	size_t capProposals = 0, capViewPool = 0, capOut = 0, capOutViewPool = 0, capCounts = 0;

	void FreeGpu(GpuDepthData& d)
	{
		if (d.d_depth)
			cudaFree(d.d_depth);
		if (d.d_normal)
			cudaFree(d.d_normal);
		if (d.d_conf)
			cudaFree(d.d_conf);
		if (d.d_depthIndex)
			cudaFree(d.d_depthIndex);
		d = GpuDepthData{};
	}

	size_t DepthDataBytes(const GpuDepthData& d) const
	{
		const size_t nPixels = (size_t)d.width * (size_t)d.height;
		return nPixels * (sizeof(float) + sizeof(uint32_t) +
			(d.d_normal ? 3 * sizeof(float) : 0) +
			(d.d_conf ? sizeof(float) : 0));
	}

	bool StoreDepthState(IIndex idxImage, const GpuDepthData& d)
	{
		const size_t nPixels = (size_t)d.width * (size_t)d.height;
		std::vector<float> depths(nPixels);
		std::vector<uint32_t> claims(nPixels);
		cudaError_t status = cudaMemcpy(depths.data(), d.d_depth, nPixels * sizeof(float), cudaMemcpyDeviceToHost);
		if (status == cudaSuccess)
			status = cudaMemcpy(claims.data(), d.d_depthIndex, nPixels * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			VERBOSE("CUDA fusion: failed to preserve cached image %u before eviction: %s",
			        idxImage, cudaGetErrorString(status));
			return false;
		}

		HostDepthState state;
		state.claimed.assign((nPixels + 7) / 8, 0);
		state.invalid.assign((nPixels + 7) / 8, 0);
		for (size_t i = 0; i < nPixels; ++i) {
			const uint8_t mask = (uint8_t)(1u << (i & 7));
			if (claims[i] != 0xFFFFFFFFu)
				state.claimed[i >> 3] |= mask;
			if (depths[i] == 0.f)
				state.invalid[i >> 3] |= mask;
		}
		hostDepthState.insert_or_assign(idxImage, std::move(state));
		return true;
	}

	bool TrimCache(size_t additionalBytes)
	{
		if (additionalBytes > maxMemory)
			return false;
		while (usedMemory > maxMemory - additionalBytes ||
		       temporaryMemory > maxMemory - additionalBytes - usedMemory) {
			auto itLRU = lru.rbegin();
			while (itLRU != lru.rend() && pinnedImages.count(*itLRU))
				++itLRU;
			if (itLRU == lru.rend())
				return false;
			const IIndex idxImage = *itLRU;
			auto it = cache.find(idxImage);
			if (it != cache.end()) {
				if (touchedImages.count(idxImage) && !StoreDepthState(idxImage, it->second))
					return false;
				usedMemory -= DepthDataBytes(it->second);
				FreeGpu(it->second);
				cache.erase(it);
			}
			touchedImages.erase(idxImage);
			lru.remove(idxImage);
		}
		return true;
	}

	template <typename T>
	bool EnsureBuffer(T*& pBuffer, size_t& capacity, size_t required, const char* name)
	{
		if (required <= capacity)
			return true;
		if (pBuffer) {
			cudaFree(pBuffer);
			temporaryMemory -= capacity * sizeof(T);
			pBuffer = nullptr;
			capacity = 0;
		}
		if (required > std::numeric_limits<size_t>::max() / sizeof(T) ||
		    !TrimCache(required * sizeof(T))) {
			VERBOSE("CUDA fusion: memory budget cannot fit %s buffer", name);
			return false;
		}
		if (cudaMalloc(&pBuffer, required * sizeof(T)) != cudaSuccess) {
			VERBOSE("CUDA fusion: failed to allocate %s buffer", name);
			return false;
		}
		capacity = required;
		temporaryMemory += capacity * sizeof(T);
		return true;
	}

	bool EnsureTempCap(size_t nPix, size_t nViews)
	{
		return EnsureBuffer(d_proposals, capProposals, nPix, "proposal") &&
		       EnsureBuffer(d_viewPool, capViewPool, nViews, "view-pool") &&
		       EnsureBuffer(d_out, capOut, nPix, "output proposal") &&
		       EnsureBuffer(d_outViewPool, capOutViewPool, nViews, "output view-pool") &&
		       EnsureBuffer(d_counts, capCounts, 6, "fusion counters");
	}
};

// ------------------------- ctor / dtor -------------------------

FuseDepthMapsCUDA::FuseDepthMapsCUDA() :
    impl_(new Impl()) {}
FuseDepthMapsCUDA::~FuseDepthMapsCUDA()
{
	Release();
	delete impl_;
}

bool FuseDepthMapsCUDA::Init()
{
	if (impl_->inited)
		return true;
	int dev = 0;
	if (cudaGetDevice(&dev) != cudaSuccess)
		return false;
	cudaDeviceProp p;
	if (cudaGetDeviceProperties(&p, dev) != cudaSuccess)
		return false;
	impl_->inited = true;
	return true;
}

void FuseDepthMapsCUDA::Release()
{
	if (!impl_)
		return;
	std::lock_guard<std::mutex> lk(impl_->mtx);
	for (auto& kv : impl_->cache)
		impl_->FreeGpu(kv.second);
	impl_->cache.clear();
	impl_->hostDepthState.clear();
	impl_->outputObservations.clear();
	impl_->touchedImages.clear();
	impl_->pinnedImages.clear();
	impl_->lru.clear();

	if (impl_->d_proposals)
		cudaFree(impl_->d_proposals);
	if (impl_->d_viewPool)
		cudaFree(impl_->d_viewPool);
	if (impl_->d_out)
		cudaFree(impl_->d_out);
	if (impl_->d_outViewPool)
		cudaFree(impl_->d_outViewPool);
	if (impl_->d_counts)
		cudaFree(impl_->d_counts);

	impl_->d_proposals = nullptr;
	impl_->d_viewPool = nullptr;
	impl_->d_out = nullptr;
	impl_->d_outViewPool = nullptr;
	impl_->d_counts = nullptr;
	impl_->capProposals = 0;
	impl_->capViewPool = 0;
	impl_->capOut = 0;
	impl_->capOutViewPool = 0;
	impl_->capCounts = 0;
	impl_->usedMemory = 0;
	impl_->temporaryMemory = 0;
	impl_->inited = false;
}

// ------------------------- upload / query -------------------------

bool FuseDepthMapsCUDA::IsCached(IIndex idxImage) const
{
	std::lock_guard<std::mutex> lk(impl_->mtx);
	return impl_->cache.count(idxImage) > 0;
}

bool FuseDepthMapsCUDA::UploadDepthData(
    IIndex idxImage, const DepthMap& depthMap, const NormalMap& normalMap,
    const ConfidenceMap& confMap, const Camera& camera, bool bNeedNormal)
{
	std::lock_guard<std::mutex> lk(impl_->mtx);

	auto it = impl_->cache.find(idxImage);
	if (it != impl_->cache.end()) {
		impl_->lru.remove(idxImage);
		impl_->lru.push_front(idxImage);
		return true;
	}
	if (depthMap.empty())
		return false;

	GpuDepthData g{};
	g.width = depthMap.cols;
	g.height = depthMap.rows;
	const size_t nPix = (size_t)g.width * (size_t)g.height;
	const bool bHasNormal = bNeedNormal && !normalMap.empty();
	const bool bHasConfidence = !confMap.empty();
	const size_t bytesPerPixel = sizeof(float) + sizeof(uint32_t) +
		(bHasNormal ? 3 * sizeof(float) : 0) +
		(bHasConfidence ? sizeof(float) : 0);
	if (nPix > std::numeric_limits<size_t>::max() / bytesPerPixel ||
	    !impl_->TrimCache(nPix * bytesPerPixel)) {
		VERBOSE("CUDA fusion: depth-map exceeds the available GPU memory budget");
		return false;
	}

	auto stateIt = impl_->hostDepthState.find(idxImage);
	std::vector<float> restoredDepth;
	std::vector<uint32_t> restoredClaims;
	if (stateIt != impl_->hostDepthState.end()) {
		const size_t bitsetSize = (nPix + 7) / 8;
		if (stateIt->second.invalid.size() != bitsetSize ||
		    stateIt->second.claimed.size() != bitsetSize) {
			VERBOSE("CUDA fusion: saved depth state size mismatch for image %u", idxImage);
			return false;
		}
		restoredDepth.resize(nPix);
		for (int y = 0; y < depthMap.rows; ++y)
			std::copy(depthMap.cv::Mat::ptr<Depth>(y),
			          depthMap.cv::Mat::ptr<Depth>(y) + depthMap.cols,
			          restoredDepth.begin() + (size_t)y * depthMap.cols);
		restoredClaims.assign(nPix, 0xFFFFFFFFu);
		for (size_t i = 0; i < nPix; ++i) {
			const uint8_t mask = (uint8_t)(1u << (i & 7));
			if (stateIt->second.invalid[i >> 3] & mask)
				restoredDepth[i] = 0.f;
			if (stateIt->second.claimed[i >> 3] & mask)
				restoredClaims[i] = 0;
		}
	}

	auto allocCopy = [&](const void* src, void** dst, size_t bytes) -> bool {
		cudaError_t status = cudaMalloc(dst, bytes);
		if (status != cudaSuccess) {
			VERBOSE("CUDA fusion: depth-map allocation failed: %s", cudaGetErrorString(status));
			return false;
		}
		if (src && (status = cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
			VERBOSE("CUDA fusion: depth-map upload failed: %s", cudaGetErrorString(status));
			return false;
		}
		return true;
	};

	const void* pDepth = restoredDepth.empty()
		? static_cast<const void*>(depthMap.ptr())
		: static_cast<const void*>(restoredDepth.data());
	if (!allocCopy(pDepth, (void**)&g.d_depth, nPix * sizeof(float))) {
		impl_->FreeGpu(g);
		return false;
	}
	if (bHasNormal) {
		if (!allocCopy(normalMap.ptr(), (void**)&g.d_normal, nPix * 3 * sizeof(float))) {
			impl_->FreeGpu(g);
			return false;
		}
	}
	if (bHasConfidence) {
		if (!allocCopy(confMap.ptr(), (void**)&g.d_conf, nPix * sizeof(float))) {
			impl_->FreeGpu(g);
			return false;
		}
	}
	cudaError_t status = cudaMalloc(&g.d_depthIndex, nPix * sizeof(uint32_t));
	if (status != cudaSuccess) {
		VERBOSE("CUDA fusion: depth-pixel claim allocation failed: %s", cudaGetErrorString(status));
		impl_->FreeGpu(g);
		return false;
	}
	status = restoredClaims.empty()
		? cudaMemset(g.d_depthIndex, 0xFF, nPix * sizeof(uint32_t))
		: cudaMemcpy(g.d_depthIndex, restoredClaims.data(),
		             nPix * sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		VERBOSE("CUDA fusion: depth-pixel claim initialization failed: %s", cudaGetErrorString(status));
		impl_->FreeGpu(g);
		return false;
	}

	for (int i = 0; i < 9; ++i) {
		g.K[i] = (float)camera.K(i / 3, i % 3);
		g.R[i] = (float)camera.R(i / 3, i % 3);
	}
	g.C[0] = (float)camera.C.x;
	g.C[1] = (float)camera.C.y;
	g.C[2] = (float)camera.C.z;

	impl_->usedMemory += impl_->DepthDataBytes(g);
	impl_->cache.emplace(idxImage, g);
	impl_->lru.push_front(idxImage);
	if (stateIt != impl_->hostDepthState.end()) {
		impl_->hostDepthState.erase(stateIt);
		impl_->touchedImages.insert(idxImage);
	}
	return true;
}

void FuseDepthMapsCUDA::SetPinnedImages(const IIndexArr& idxImages)
{
	std::lock_guard<std::mutex> lk(impl_->mtx);
	impl_->pinnedImages.clear();
	for (const IIndex idxImage : idxImages)
		impl_->pinnedImages.insert(idxImage);
	impl_->TrimCache(0);
}

bool FuseDepthMapsCUDA::PrepareFusionBuffers(size_t nPixels, size_t nNeighbors)
{
	if (nPixels == 0 || nPixels > (size_t)INT_MAX ||
	    nNeighbors >= UINT32_MAX ||
	    nPixels > UINT32_MAX / (nNeighbors + 1))
		return false;
	const size_t nViewSlots = nPixels * (nNeighbors + 1);
	std::lock_guard<std::mutex> lk(impl_->mtx);
	return impl_->EnsureTempCap(nPixels, nViewSlots);
}

void FuseDepthMapsCUDA::SetMaxMemory(size_t b)
{
	std::lock_guard<std::mutex> lk(impl_->mtx);
	impl_->maxMemory = b;
	impl_->TrimCache(0);
}

size_t FuseDepthMapsCUDA::GetUsedMemory() const
{
	std::lock_guard<std::mutex> lk(impl_->mtx);
	return impl_->usedMemory + impl_->temporaryMemory;
}

// ------------------------- FuseOne -------------------------

bool FuseDepthMapsCUDA::FuseOne(
    IIndex idxImage, const IIndexArr& neighbors,
    unsigned nMinViewsFuse, float depthThresh, float normalCosThresh,
    bool bEstimateColor, bool bEstimateNormal,
    const ImageArr& images, PointCloud& pointcloud)
{
	if (!impl_->inited && !Init())
		return false;
	cudaStream_t stream = 0;

	// Grab a reference to the reference depth-map.
	GpuDepthData* pRef = nullptr;
	{
		std::lock_guard<std::mutex> lk(impl_->mtx);
		auto it = impl_->cache.find(idxImage);
		if (it == impl_->cache.end())
			return false;
		pRef = &it->second;
		impl_->lru.remove(idxImage);
		impl_->lru.push_front(idxImage);
	}

	// Preserve the depth-pixel claim maps to prevent duplicate points across references.
	std::vector<GpuNeighborView> hNb;
	hNb.reserve(neighbors.size());
	for (IIndex nid : neighbors) {
		std::lock_guard<std::mutex> lk(impl_->mtx);
		auto it = impl_->cache.find(nid);
		if (it == impl_->cache.end() || !it->second.d_depth)
			continue;
		GpuNeighborView v;
		v.data = it->second;
		v.idxImage = nid;
		hNb.push_back(v);
	}
	if (hNb.empty())
		return false;

	const int N = (int)hNb.size();
	const int W = pRef->width, H = pRef->height;
	const size_t nPix = (size_t)W * (size_t)H;
	if (nPix > (size_t)INT_MAX || nPix > UINT32_MAX / (size_t)(N + 1))
		return false;
	const size_t nViewSlots = nPix * (size_t)(N + 1);
	if (nViewSlots > UINT32_MAX)
		return false;

	{
		std::lock_guard<std::mutex> lk(impl_->mtx);
		if (!impl_->TrimCache((size_t)N * sizeof(GpuNeighborView)) ||
		    !impl_->EnsureTempCap(nPix, nViewSlots))
			return false;
	}

	GpuNeighborView* d_nb = nullptr;
	if (cudaMalloc(&d_nb, (size_t)N * sizeof(GpuNeighborView)) != cudaSuccess) {
		VERBOSE("CUDA fusion: failed to allocate neighbor descriptors");
		return false;
	}
	if (cudaMemcpy(d_nb, hNb.data(), N * sizeof(GpuNeighborView), cudaMemcpyHostToDevice) != cudaSuccess) {
		VERBOSE("CUDA fusion: failed to upload neighbor descriptors");
		cudaFree(d_nb);
		return false;
	}
	cudaError_t cudaStatus = cudaMemsetAsync(impl_->d_counts, 0, 6 * sizeof(uint32_t), stream);
	if (cudaStatus != cudaSuccess) {
		VERBOSE("CUDA fusion: failed to initialize diagnostic counters: %s", cudaGetErrorString(cudaStatus));
		cudaFree(d_nb);
		return false;
	}

	// Propose, atomically claim each participating depth pixel, invalidate
	// conflicting unclaimed depths, then compact the accepted outputs.
	launchFuseProposal(*pRef, d_nb, N, depthThresh, normalCosThresh, nMinViewsFuse,
	                   impl_->d_proposals, impl_->d_viewPool, impl_->d_counts,
	                   W, H, stream);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		VERBOSE("CUDA fusion kernel failed: %s", cudaGetErrorString(cudaStatus));
		cudaFree(d_nb);
		return false;
	}

	launchClaimViews(*pRef, impl_->d_proposals,
	                 impl_->d_viewPool, d_nb, N, (int)nPix,
	                 impl_->d_counts, stream);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		VERBOSE("CUDA depth-pixel claim kernel failed: %s", cudaGetErrorString(cudaStatus));
		cudaFree(d_nb);
		return false;
	}

	launchApplyInvalid(*pRef, d_nb, N, impl_->d_proposals,
	                   depthThresh, normalCosThresh, impl_->d_counts, stream);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		VERBOSE("CUDA invalid-depth kernel failed: %s", cudaGetErrorString(cudaStatus));
		cudaFree(d_nb);
		return false;
	}

	launchSelectValid(impl_->d_proposals, (int)nPix, impl_->d_viewPool,
	                  impl_->d_out, impl_->d_outViewPool, impl_->d_counts, stream);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		VERBOSE("CUDA fusion selection failed: %s", cudaGetErrorString(cudaStatus));
		cudaFree(d_nb);
		return false;
	}

	uint32_t counts[6] = {};
	cudaStatus = cudaMemcpy(counts, impl_->d_counts, sizeof(counts), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		VERBOSE("CUDA fusion: failed to download selection counters: %s", cudaGetErrorString(cudaStatus));
		cudaFree(d_nb);
		return false;
	}
	{
		std::lock_guard<std::mutex> lk(impl_->mtx);
		impl_->touchedImages.insert(idxImage);
		for (const GpuNeighborView& neighbor : hNb)
			impl_->touchedImages.insert(neighbor.idxImage);
	}
	if (counts[0] > nPix || counts[1] > nViewSlots) {
		VERBOSE("CUDA fusion: invalid selection counters");
		cudaFree(d_nb);
		return false;
	}

	if (counts[0] > 0) {
		std::vector<FusionProposal> hOut(counts[0]);
		if (cudaMemcpy(hOut.data(), impl_->d_out, (size_t)counts[0] * sizeof(FusionProposal), cudaMemcpyDeviceToHost) != cudaSuccess) {
			VERBOSE("CUDA fusion: failed to download proposals");
			cudaFree(d_nb);
			return false;
		}

		std::vector<ViewEntry> hPool(counts[1]);
		if (cudaMemcpy(hPool.data(), impl_->d_outViewPool,
		               (size_t)counts[1] * sizeof(ViewEntry), cudaMemcpyDeviceToHost) != cudaSuccess) {
			VERBOSE("CUDA fusion: failed to download view-pool entries");
			cudaFree(d_nb);
			return false;
		}

		const Image& refImage = images[idxImage];
		std::vector<SEACAVE::PairIdx> observations;
		observations.reserve(counts[1]);
		size_t duplicateObservations = 0;
		size_t pointsWithOneView = 0;
		size_t pointsWithTwoViews = 0;
		size_t pointsWithThreeViews = 0;
		size_t pointsWithFourOrMoreViews = 0;
		for (const auto& p : hOut) {
			if (p.viewBegin > hPool.size() || p.viewCount > hPool.size() - p.viewBegin) {
				VERBOSE("CUDA fusion: proposal contains an invalid view-pool range");
				cudaFree(d_nb);
				return false;
			}
			pointcloud.points.push_back(Point3(p.point[0], p.point[1], p.point[2]));

			PointCloud::ViewArr views;
			PointCloud::WeightArr weights;
			views.Reserve(p.viewCount);
			weights.Reserve(p.viewCount);
			for (uint32_t k = 0; k < p.viewCount; ++k) {
				const ViewEntry& entry = hPool[p.viewBegin + k];
				if (entry.neighborIndex != NO_ID && entry.neighborIndex >= hNb.size()) {
					VERBOSE("CUDA fusion: proposal contains an invalid neighbor index");
					cudaFree(d_nb);
					return false;
				}
				const IIndex idxView = entry.neighborIndex == NO_ID
					? idxImage : hNb[entry.neighborIndex].idxImage;
				observations.emplace_back(idxView, entry.pixelIndex);
				const IIndex position = views.InsertSort(idxView);
				weights.InsertAt(position, entry.weight);
			}
			if (p.viewCount == 1)
				++pointsWithOneView;
			else if (p.viewCount == 2)
				++pointsWithTwoViews;
			else if (p.viewCount == 3)
				++pointsWithThreeViews;
			else if (p.viewCount >= 4)
				++pointsWithFourOrMoreViews;
			pointcloud.pointViews.push_back(views);
			pointcloud.pointWeights.push_back(weights);

			if (bEstimateNormal)
				pointcloud.normals.push_back(Point3(p.normal[0], p.normal[1], p.normal[2]));

			if (bEstimateColor) {
				// The CUDA path currently uses the reference pixel color.
				pointcloud.colors.push_back(refImage.image(p.y, p.x));
			}
		}
		{
			std::lock_guard<std::mutex> lk(impl_->mtx);
			for (const SEACAVE::PairIdx& observation : observations)
				if (!impl_->outputObservations.emplace(observation).second)
					++duplicateObservations;
		}
		VERBOSE("CUDA fusion stats image %u: neighbors=%d proposals=%u claim_success=%u claim_failures=%u rolled_back_pixels=%u occluded_depths_removed=%u output_points=%u views[1/2/3/4+]=%llu/%llu/%llu/%llu duplicate_observations=%llu",
		        idxImage, N, counts[2], counts[2] - counts[3], counts[3], counts[4], counts[5], counts[0],
		        (unsigned long long)pointsWithOneView,
		        (unsigned long long)pointsWithTwoViews,
		        (unsigned long long)pointsWithThreeViews,
		        (unsigned long long)pointsWithFourOrMoreViews,
		        (unsigned long long)duplicateObservations);
	} else {
		VERBOSE("CUDA fusion stats image %u: neighbors=%d proposals=%u claim_success=%u claim_failures=%u rolled_back_pixels=%u occluded_depths_removed=%u output_points=0 views[1/2/3/4+]=0/0/0/0 duplicate_observations=0",
		        idxImage, N, counts[2], counts[2] - counts[3], counts[3], counts[4], counts[5]);
	}

	cudaFree(d_nb);
	return true;
}

#endif // _USE_CUDA