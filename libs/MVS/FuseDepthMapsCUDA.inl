/*
* FuseDepthMapsCUDA.inl
*
* Light-weight declarations shared by both host (.cpp) and device (.cu)
* compilation units. MUST NOT include any openMVS host-only header
* (Common.h / DepthMap.h / SceneDensify.h ...) so that nvcc sees a clean
* translation unit.
*/

#ifndef _MVS_FUSEDEPTHMAPSCUDA_INL_
#define _MVS_FUSEDEPTHMAPSCUDA_INL_


	#include <cuda_runtime.h>
	#include <cstdint>
	#include <cstddef>

	// IIndex / NO_ID are normally provided by Common.h on the host side.
	// For the .cu TU (which does not include Common.h), provide fallbacks.
#ifndef _MVS_COMMON_H_
typedef uint32_t IIndex;
	#ifndef NO_ID
		#define NO_ID 0xFFFFFFFFu
	#endif
#endif

namespace MVS {

// Device-side (GPU) copy of one depth-map's data. POD; safe to pass to kernels.
struct GpuDepthData
{
	float* d_depth = nullptr; // W*H, 0 = invalid
	float* d_normal = nullptr; // 3*W*H (Point3f layout), may be null
	float* d_conf = nullptr; // W*H, may be null (treated as 1.0)
	uint32_t* d_depthIndex = nullptr; // W*H, dedup table, 0xFFFFFFFF = free
	int width = 0;
	int height = 0;
	float K[9] = {0}; // row-major
	float R[9] = {0}; // world->camera, row-major
	float C[3] = {0};

	__host__ __device__ inline bool IsValid() const { return d_depth != nullptr; }
	__host__ __device__ inline size_t NumPixels() const { return (size_t)width * (size_t)height; }
};

// Light-weight neighbor handle: copies GpuDepthData but does NOT own memory.
struct GpuNeighborView
{
	GpuDepthData data;
	IIndex idxImage = NO_ID;
};

// Pass1 output: one fusion proposal per reference pixel.
struct FusionProposal
{
	float point[3];
	float normal[3];
	float confidence;
	uint32_t viewBegin;
	uint32_t viewCount;
	uint16_t x, y;
	uint8_t valid;
	uint8_t pad[1];
};

// Entry in the per-fusion view pool.
struct ViewEntry
{
	IIndex idxImage;
	float weight;
};

// ---------------------------------------------------------------------------
// Launchers implemented in FuseDepthMapsCUDA.cu (plain C++ signatures so that
// the host .cpp can call them without seeing any CUDA syntax).
// ---------------------------------------------------------------------------

void launchFuseProposal(
    const GpuDepthData& ref,
    GpuNeighborView* d_neighbors,
    int N,
    float depthThresh,
    float normalCosThresh,
    unsigned nMinViewsFuse,
    FusionProposal* d_proposals,
    ViewEntry* d_viewPool,
    uint32_t* d_viewPoolCounter,
    unsigned refIdx,
    int W,
    int H,
    cudaStream_t stream);

void launchApplyInvalid(
    const GpuDepthData& ref,
    GpuNeighborView* d_neighbors,
    int N,
    const FusionProposal* d_proposals,
    float depthThresh,
    float normalCosThresh,
    cudaStream_t stream);

void launchSelectValid(
    const FusionProposal* d_in,
    int n,
    FusionProposal* d_out,
    int* d_numOut,
    cudaStream_t stream);

} // namespace MVS

#endif // _MVS_FUSEDEPTHMAPSCUDA_INL_