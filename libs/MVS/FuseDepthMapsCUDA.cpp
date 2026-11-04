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
	#include <unordered_map>
	#include <list>
	#include <mutex>
	#include <vector>

using namespace MVS;

// ------------------------- Impl -------------------------

struct FuseDepthMapsCUDA::Impl
{
	std::unordered_map<IIndex, GpuDepthData> cache;
	std::list<IIndex> lru; // front = most recently used
	mutable std::mutex mtx;
	size_t maxMemory = 2ull * 1024 * 1024 * 1024; // 2 GB
	size_t usedMemory = 0;
	bool inited = false;

	FusionProposal* d_proposals = nullptr;

	ViewEntry* d_viewPool = nullptr;
	uint32_t* d_viewPoolCounter = nullptr;
	int2* d_invalidDepths = nullptr;
	uint32_t* d_invalidCount = nullptr;
	size_t capProposals = 0, capViewPool = 0, capInvalid = 0;

	void FreeGpu(GpuDepthData& d)
	{
		if (d.d_depth)
			cudaFree(d.d_depth);
		if (d.d_normal)
			cudaFree(d.d_normal);
		if (d.d_conf)
			cudaFree(d.d_conf);
		if (d.ownsDepthIndex && d.d_depthIndex)
			cudaFree(d.d_depthIndex);
		d = GpuDepthData{};
	}

	bool EnsureTempCap(size_t nPix, size_t nViews, size_t nInvalid)
	{
		if (nPix > capProposals) {
			if (d_proposals)
				cudaFree(d_proposals);
			d_proposals = nullptr;
			capProposals = 0;
			if (cudaMalloc(&d_proposals, nPix * sizeof(FusionProposal)) != cudaSuccess) {
				VERBOSE("CUDA fusion: failed to allocate proposal buffer");
				return false;
			}
			capProposals = nPix;
		}
		if (nViews > capViewPool) {
			if (d_viewPool)
				cudaFree(d_viewPool);
			if (d_viewPoolCounter)
				cudaFree(d_viewPoolCounter);
			d_viewPool = nullptr;
			d_viewPoolCounter = nullptr;
			capViewPool = 0;
			if (cudaMalloc(&d_viewPool, nViews * sizeof(ViewEntry)) != cudaSuccess) {
				VERBOSE("CUDA fusion: failed to allocate view-pool buffer");
				return false;
			}
			if (cudaMalloc(&d_viewPoolCounter, sizeof(uint32_t)) != cudaSuccess) {
				cudaFree(d_viewPool);
				d_viewPool = nullptr;
				VERBOSE("CUDA fusion: failed to allocate view-pool counter");
				return false;
			}
			capViewPool = nViews;
		}
		if (nInvalid > capInvalid) {
			if (d_invalidDepths)
				cudaFree(d_invalidDepths);
			if (d_invalidCount)
				cudaFree(d_invalidCount);
			d_invalidDepths = nullptr;
			d_invalidCount = nullptr;
			capInvalid = 0;
			if (cudaMalloc(&d_invalidDepths, nInvalid * sizeof(int2)) != cudaSuccess) {
				VERBOSE("CUDA fusion: failed to allocate invalid-depth buffer");
				return false;
			}
			if (cudaMalloc(&d_invalidCount, sizeof(uint32_t)) != cudaSuccess) {
				cudaFree(d_invalidDepths);
				d_invalidDepths = nullptr;
				VERBOSE("CUDA fusion: failed to allocate invalid-depth counter");
				return false;
			}
			capInvalid = nInvalid;
		}
		return true;
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
	impl_->lru.clear();

	if (impl_->d_proposals)
		cudaFree(impl_->d_proposals);
	if (impl_->d_viewPool)
		cudaFree(impl_->d_viewPool);
	if (impl_->d_viewPoolCounter)
		cudaFree(impl_->d_viewPoolCounter);
	if (impl_->d_invalidDepths)
		cudaFree(impl_->d_invalidDepths);
	if (impl_->d_invalidCount)
		cudaFree(impl_->d_invalidCount);

	impl_->d_proposals = nullptr;
	impl_->d_viewPool = nullptr;
	impl_->d_viewPoolCounter = nullptr;
	impl_->d_invalidDepths = nullptr;
	impl_->d_invalidCount = nullptr;
	impl_->capProposals = 0;
	impl_->capViewPool = 0;
	impl_->capInvalid = 0;
	impl_->usedMemory = 0;
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
	g.ownsDepthIndex = true;
	const size_t nPix = (size_t)g.width * (size_t)g.height;

	auto allocCopy = [&](const void* src, void** dst, size_t bytes) -> bool {
		if (cudaMalloc(dst, bytes) != cudaSuccess)
			return false;
		if (src && cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
			return false;
		return true;
	};

	if (!allocCopy(depthMap.ptr(), (void**)&g.d_depth, nPix * sizeof(float))) {
		impl_->FreeGpu(g);
		return false;
	}
	if (bNeedNormal && !normalMap.empty()) {
		if (!allocCopy(normalMap.ptr(), (void**)&g.d_normal, nPix * 3 * sizeof(float))) {
			impl_->FreeGpu(g);
			return false;
		}
	}
	if (!confMap.empty()) {
		if (!allocCopy(confMap.ptr(), (void**)&g.d_conf, nPix * sizeof(float))) {
			impl_->FreeGpu(g);
			return false;
		}
	}
	if (cudaMalloc(&g.d_depthIndex, nPix * sizeof(uint32_t)) != cudaSuccess) {
		impl_->FreeGpu(g);
		return false;
	}
	if (cudaMemset(g.d_depthIndex, 0xFF, nPix * sizeof(uint32_t)) != cudaSuccess) {
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

	impl_->usedMemory += nPix * (4 + (bNeedNormal ? 12 : 0) + 4 + 4);
	impl_->cache.emplace(idxImage, g);
	impl_->lru.push_front(idxImage);
	return true;
}

void FuseDepthMapsCUDA::SetMaxMemory(size_t b)
{
	std::lock_guard<std::mutex> lk(impl_->mtx);
	impl_->maxMemory = b;
}

size_t FuseDepthMapsCUDA::GetUsedMemory() const { return impl_->usedMemory; }

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

	// Build neighbor list (depthIndex disabled on neighbors: per-reference
	// fusion is independent, so we don't need cross-fusion dedup).
	std::vector<GpuNeighborView> hNb;
	hNb.reserve(neighbors.size());
	for (IIndex nid : neighbors) {
		std::lock_guard<std::mutex> lk(impl_->mtx);
		auto it = impl_->cache.find(nid);
		if (it == impl_->cache.end() || !it->second.d_depth)
			continue;
		GpuNeighborView v;
		v.data = it->second;
		v.data.d_depthIndex = nullptr;
		v.idxImage = nid;
		hNb.push_back(v);
	}
	if (hNb.empty())
		return false;

	const int N = (int)hNb.size();
	const int W = pRef->width, H = pRef->height;
	const size_t nPix = (size_t)W * (size_t)H;
	if (nPix > UINT32_MAX / (size_t)(N + 1))
		return false;
	// Each proposal can consume the reference view plus up to N neighbors, so the
	// view pool must be sized for the worst-case slot count per pixel, not merely
	// the minimum number of fused views.
	const size_t nViewSlots = nPix * (size_t)(N + 1);
	const size_t nInvalidSlots = nPix * (size_t)N;
	if (nInvalidSlots > UINT32_MAX ||
	    !impl_->EnsureTempCap(nPix, nViewSlots, nInvalidSlots))
		return false;

	GpuNeighborView* d_nb = nullptr;
	if (cudaMalloc(&d_nb, N * sizeof(GpuNeighborView)) != cudaSuccess)
		return false;
	if (cudaMemcpy(d_nb, hNb.data(), N * sizeof(GpuNeighborView), cudaMemcpyHostToDevice) != cudaSuccess) {
		cudaFree(d_nb);
		return false;
	}

	if (cudaMemset(impl_->d_viewPoolCounter, 0, sizeof(uint32_t)) != cudaSuccess ||
	    cudaMemset(impl_->d_invalidCount, 0, sizeof(uint32_t)) != cudaSuccess ||
	    cudaMemset(pRef->d_depthIndex, 0xFF, nPix * sizeof(uint32_t)) != cudaSuccess) {
		cudaFree(d_nb);
		return false;
	}

	// ---- Pass 1 ----
	launchFuseProposal(*pRef, d_nb, N, depthThresh, normalCosThresh, nMinViewsFuse,
	                   impl_->d_proposals, impl_->d_viewPool, impl_->d_viewPoolCounter,
	                   impl_->d_invalidDepths, impl_->d_invalidCount,
	                   (uint32_t)nInvalidSlots,
	                   (unsigned)idxImage, W, H, stream);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus == cudaSuccess)
		cudaStatus = cudaStreamSynchronize(stream);
	if (cudaStatus != cudaSuccess) {
		VERBOSE("CUDA fusion kernel failed: %s", cudaGetErrorString(cudaStatus));
		cudaFree(d_nb);
		return false;
	}

	uint32_t nInvalid = 0;
	if (cudaMemcpy(&nInvalid, impl_->d_invalidCount, sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
		cudaFree(d_nb);
		return false;
	}
	nInvalid = (uint32_t)MINF(nInvalid, (uint32_t)nInvalidSlots);
	launchApplyInvalid(impl_->d_invalidDepths, nInvalid, d_nb, N, stream);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		VERBOSE("CUDA invalid-depth kernel failed: %s", cudaGetErrorString(cudaStatus));
		cudaFree(d_nb);
		return false;
	}

	FusionProposal* d_out = nullptr;
	int* d_numOut = nullptr;
	if (cudaMalloc(&d_out, nPix * sizeof(FusionProposal)) != cudaSuccess) {
		cudaFree(d_nb);
		return false;
	}
	if (cudaMalloc(&d_numOut, sizeof(int)) != cudaSuccess) {
		cudaFree(d_out);
		cudaFree(d_nb);
		return false;
	}

	launchSelectValid(impl_->d_proposals, (int)nPix, d_out, d_numOut, stream);

	int numSel = 0;
	cudaStatus = cudaGetLastError();
	if (cudaStatus == cudaSuccess)
		cudaStatus = cudaMemcpy(&numSel, d_numOut, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		VERBOSE("CUDA fusion selection failed: %s", cudaGetErrorString(cudaStatus));
		cudaFree(d_out);
		cudaFree(d_numOut);
		cudaFree(d_nb);
		return false;
	}

	if (numSel > 0) {
		std::vector<FusionProposal> hOut((size_t)numSel);
		if (cudaMemcpy(hOut.data(), d_out, (size_t)numSel * sizeof(FusionProposal), cudaMemcpyDeviceToHost) != cudaSuccess) {
			VERBOSE("CUDA fusion: failed to download proposals");
			cudaFree(d_out);
			cudaFree(d_numOut);
			cudaFree(d_nb);
			return false;
		}

		// Read the view pool. Conservative: copy the whole capacity.
		std::vector<ViewEntry> hPool(impl_->capViewPool);
		if (cudaMemcpy(hPool.data(), impl_->d_viewPool,
		               impl_->capViewPool * sizeof(ViewEntry), cudaMemcpyDeviceToHost) != cudaSuccess) {
			VERBOSE("CUDA fusion: failed to download view-pool entries");
			cudaFree(d_out);
			cudaFree(d_numOut);
			cudaFree(d_nb);
			return false;
		}

		const Image& refImage = images[idxImage];
		for (const auto& p : hOut) {
			if (p.viewBegin > hPool.size() || p.viewCount > hPool.size() - p.viewBegin) {
				VERBOSE("CUDA fusion: proposal contains an invalid view-pool range");
				cudaFree(d_out);
				cudaFree(d_numOut);
				cudaFree(d_nb);
				return false;
			}
			pointcloud.points.push_back(Point3(p.point[0], p.point[1], p.point[2]));

			PointCloud::ViewArr views;
			PointCloud::WeightArr weights;
			views.Reserve(p.viewCount);
			weights.Reserve(p.viewCount);
			for (uint32_t k = 0; k < p.viewCount; ++k) {
				views.push_back(hPool[p.viewBegin + k].idxImage);
				weights.push_back(hPool[p.viewBegin + k].weight);
			}
			pointcloud.pointViews.push_back(views);
			pointcloud.pointWeights.push_back(weights);

			if (bEstimateNormal)
				pointcloud.normals.push_back(Point3(p.normal[0], p.normal[1], p.normal[2]));

			if (bEstimateColor) {
				// Simple: take the reference pixel color (CPU fusion averages
				// over all fused views).
				pointcloud.colors.push_back(refImage.image(p.y, p.x));
			}
		}
	}

	cudaFree(d_out);
	cudaFree(d_numOut);
	cudaFree(d_nb);
	return true;
}

#endif // _USE_CUDA