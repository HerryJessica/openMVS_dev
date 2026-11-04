/*
* FuseDepthMapsCUDA.cu
*
* CUDA kernels + launchers for GPU fusion. Compiled by NVCC only.
* No CCCL/cub dependency: compaction is done by a hand-written kernel.
*
* IMPORTANT: only include the light-weight .inl (POD structs + launcher
* declarations). Do NOT include FuseDepthMapsCUDA.h / Common.h / DepthMap.h,
* otherwise nvcc front-end parses the heavy host headers and CUDA device
* intrinsics (atomicAdd, blockIdx, tex2D, __float2int_rn ...) get demoted to
* "undefined identifier".
*/

#include "FuseDepthMapsCUDA.inl"


#include <cuda_runtime.h>

namespace MVS {

// ------------------------- device helpers -------------------------

__device__ __forceinline__ bool isInside(int x, int y, int w, int h)
{
	return (unsigned)x < (unsigned)w && (unsigned)y < (unsigned)h;
}

__device__ __forceinline__ float3 transformPointI2W(
    const float K[9], const float R[9], const float C[3],
    float x, float y, float depth)
{
	const float xn = (x - K[2]) / K[0];
	const float yn = (y - K[5]) / K[4];
	const float Xc[3] = {xn * depth, yn * depth, depth};
	float3 o;
	// Xw = R^T * Xc + C
	o.x = R[0] * Xc[0] + R[3] * Xc[1] + R[6] * Xc[2] + C[0];
	o.y = R[1] * Xc[0] + R[4] * Xc[1] + R[7] * Xc[2] + C[1];
	o.z = R[2] * Xc[0] + R[5] * Xc[1] + R[8] * Xc[2] + C[2];
	return o;
}

__device__ __forceinline__ float3 projectPointW2C(
    const float K[9], const float R[9], const float C[3], float3 Xw)
{
	const float dx = Xw.x - C[0], dy = Xw.y - C[1], dz = Xw.z - C[2];
	const float Xc = R[0] * dx + R[1] * dy + R[2] * dz;
	const float Yc = R[3] * dx + R[4] * dy + R[5] * dz;
	const float Zc = R[6] * dx + R[7] * dy + R[8] * dz;
	if (Zc <= 0.f)
		return make_float3(-1.f, -1.f, Zc);
	return make_float3(K[0] * (Xc / Zc) + K[2], K[4] * (Yc / Zc) + K[5], Zc);
}

__device__ __forceinline__ bool isDepthSimilar(float a, float b, float th)
{
	if (a < 1e-6f)
		return false;
	return fabsf(a - b) / a < th;
}

__device__ __forceinline__ float conf2Weight(float conf, float depth)
{
	return 1.f / (fmaxf(1.f - conf, 0.03f) * depth * depth);
}

__device__ __forceinline__ float3 normalCam2World(const float R[9], float3 n)
{
	float3 o;
	o.x = R[0] * n.x + R[3] * n.y + R[6] * n.z;
	o.y = R[1] * n.x + R[4] * n.y + R[7] * n.z;
	o.z = R[2] * n.x + R[5] * n.y + R[8] * n.z;
	return o;
}

// ------------------------- Pass1 -------------------------

__global__ void kernel_fuseProposal(
    GpuDepthData ref,
    const GpuNeighborView* neighbors, int N,
    float depthThresh,
    float normalCosThresh,
    unsigned nMinViewsFuse,
    FusionProposal* outProposal,
    ViewEntry* viewPool,
    uint32_t* viewPoolCounter,
    unsigned refIdx)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= ref.width || y >= ref.height)
		return;
	const int idx = y * ref.width + x;

	FusionProposal& prop = outProposal[idx];
	prop.valid = 0;
	prop.viewCount = 0;
	prop.viewBegin = 0;

	const float depth = ref.d_depth[idx];
	if (depth <= 0.f)
		return;

	if (ref.d_depthIndex) {
		uint32_t prev = atomicCAS(&ref.d_depthIndex[idx], 0xFFFFFFFFu, (uint32_t)idx);
		if (prev != 0xFFFFFFFFu)
			return;
	}

	const float3 Xw = transformPointI2W(ref.K, ref.R, ref.C, (float)x, (float)y, depth);

	float3 Nw = make_float3(0.f, 0.f, -1.f);
	if (ref.d_normal) {
		const float3 nCam = make_float3(ref.d_normal[idx * 3], ref.d_normal[idx * 3 + 1], ref.d_normal[idx * 3 + 2]);
		Nw = normalCam2World(ref.R, nCam);
	}

	const float conf = conf2Weight(ref.d_conf ? ref.d_conf[idx] : 1.f, depth);
	// Reserve a complete view block for this proposal. The fused set contains
	// the reference view plus up to N neighboring views, so each proposal must
	// claim N+1 entries instead of just one; otherwise different pixels overlap
	// and overwrite each other's view records in the shared pool.
	const uint32_t begin = atomicAdd(viewPoolCounter, (uint32_t)(N + 1u));
	viewPool[begin] = {(IIndex)refIdx, conf};

	float3 Xacc = make_float3(Xw.x * conf, Xw.y * conf, Xw.z * conf);
	float3 Nacc = make_float3(Nw.x * conf, Nw.y * conf, Nw.z * conf);
	float confSum = conf;
	int viewCount = 1;

	for (int n = 0; n < N; ++n) {
		const GpuDepthData& nd = neighbors[n].data;
		if (!nd.d_depth)
			continue;

		const float3 pt = projectPointW2C(nd.K, nd.R, nd.C, Xw);
		if (pt.z <= 0.f)
			continue;
		const int xb = __float2int_rn(pt.x);
		const int yb = __float2int_rn(pt.y);
		if (!isInside(xb, yb, nd.width, nd.height))
			continue;

		const int idxB = yb * nd.width + xb;
		const float dB = nd.d_depth[idxB];
		if (dB <= 0.f)
			continue;

		bool fused = false;
		if (isDepthSimilar(pt.z, dB, depthThresh) && nd.d_normal) {
			const float3 nB = make_float3(nd.d_normal[idxB * 3], nd.d_normal[idxB * 3 + 1], nd.d_normal[idxB * 3 + 2]);
			const float3 NBw = normalCam2World(nd.R, nB);
			const float ndot = Nw.x * NBw.x + Nw.y * NBw.y + Nw.z * NBw.z;
			if (ndot > normalCosThresh) {
				const float cB = conf2Weight(nd.d_conf ? nd.d_conf[idxB] : 1.f, dB);
				const float3 XB = transformPointI2W(nd.K, nd.R, nd.C, (float)xb, (float)yb, dB);
				Xacc.x += XB.x * cB;
				Xacc.y += XB.y * cB;
				Xacc.z += XB.z * cB;
				Nacc.x += NBw.x * cB;
				Nacc.y += NBw.y * cB;
				Nacc.z += NBw.z * cB;
				confSum += cB;
				viewPool[begin + viewCount] = {neighbors[n].idxImage, cB};
				++viewCount;
				fused = true;
			}
		}
	}

	if ((unsigned)viewCount < nMinViewsFuse)
		return;

	const float inv = 1.f / confSum;
	prop.point[0] = Xacc.x * inv;
	prop.point[1] = Xacc.y * inv;
	prop.point[2] = Xacc.z * inv;
	prop.normal[0] = Nacc.x * inv;
	prop.normal[1] = Nacc.y * inv;
	prop.normal[2] = Nacc.z * inv;
	const float nl = sqrtf(prop.normal[0] * prop.normal[0] + prop.normal[1] * prop.normal[1] + prop.normal[2] * prop.normal[2]);
	if (nl > 1e-8f) {
		prop.normal[0] /= nl;
		prop.normal[1] /= nl;
		prop.normal[2] /= nl;
	}
	prop.confidence = confSum;
	prop.viewBegin = begin;
	prop.viewCount = (uint32_t)viewCount;
	prop.x = (uint16_t)x;
	prop.y = (uint16_t)y;
	prop.valid = 1;
}

__global__ void kernel_applyInvalid(
    GpuDepthData ref, GpuNeighborView* nb, int N,
    const FusionProposal* proposals, float depthThresh, float normalCosThresh)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= ref.width || y >= ref.height)
		return;
	const int idx = y * ref.width + x;
	if (!proposals[idx].valid)
		return;

	const float depth = ref.d_depth[idx];
	const float3 Xw = transformPointI2W(ref.K, ref.R, ref.C, (float)x, (float)y, depth);
	float3 Nw = make_float3(0.f, 0.f, -1.f);
	if (ref.d_normal) {
		const float3 nCam = make_float3(ref.d_normal[idx * 3], ref.d_normal[idx * 3 + 1], ref.d_normal[idx * 3 + 2]);
		Nw = normalCam2World(ref.R, nCam);
	}

	for (int n = 0; n < N; ++n) {
		GpuDepthData& nd = nb[n].data;
		if (!nd.d_depth)
			continue;
		const float3 pt = projectPointW2C(nd.K, nd.R, nd.C, Xw);
		if (pt.z <= 0.f)
			continue;
		const int xb = __float2int_rn(pt.x);
		const int yb = __float2int_rn(pt.y);
		if (!isInside(xb, yb, nd.width, nd.height))
			continue;

		const int idxB = yb * nd.width + xb;
		const float dB = nd.d_depth[idxB];
		if (dB <= 0.f || pt.z >= dB)
			continue;

		bool fused = false;
		if (isDepthSimilar(pt.z, dB, depthThresh) && nd.d_normal) {
			const float3 nB = make_float3(nd.d_normal[idxB * 3], nd.d_normal[idxB * 3 + 1], nd.d_normal[idxB * 3 + 2]);
			const float3 NBw = normalCam2World(nd.R, nB);
			fused = Nw.x * NBw.x + Nw.y * NBw.y + Nw.z * NBw.z > normalCosThresh;
		}
		if (!fused)
			atomicExch(&nd.d_depth[idxB], 0.f);
	}
}

// ------------------------- compaction (replaces cub DeviceSelect) -------------------------

__global__ void kernel_selectValid(const FusionProposal* __restrict__ in, int n,
                                   FusionProposal* __restrict__ out,
                                   int* __restrict__ numOut)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	if (in[i].valid) {
		const int pos = atomicAdd(numOut, 1);
		out[pos] = in[i];
	}
}

// ------------------------- launchers -------------------------

void launchFuseProposal(
    const GpuDepthData& ref, GpuNeighborView* d_neighbors, int N,
    float depthThresh, float normalCosThresh, unsigned nMinViewsFuse,
    FusionProposal* d_proposals, ViewEntry* d_viewPool, uint32_t* d_viewPoolCounter,
    unsigned refIdx, int W, int H, cudaStream_t stream)
{
	const dim3 block(16, 16);
	const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
	kernel_fuseProposal<<<grid, block, 0, stream>>>(
	    ref, d_neighbors, N, depthThresh, normalCosThresh, nMinViewsFuse,
	    d_proposals, d_viewPool, d_viewPoolCounter, refIdx);
}

void launchApplyInvalid(const GpuDepthData& ref, GpuNeighborView* nb, int N,
                        const FusionProposal* proposals, float depthThresh,
                        float normalCosThresh, cudaStream_t stream)
{
	const dim3 block(16, 16);
	const dim3 grid((ref.width + block.x - 1) / block.x,
	                (ref.height + block.y - 1) / block.y);
	kernel_applyInvalid<<<grid, block, 0, stream>>>(
	    ref, nb, N, proposals, depthThresh, normalCosThresh);
}

void launchSelectValid(
    const FusionProposal* d_in, int n,
    FusionProposal* d_out, int* d_numOut,
    cudaStream_t stream)
{
	// reset counter
	cudaMemsetAsync(d_numOut, 0, sizeof(int), stream);
	if (n <= 0)
		return;
	const int blk = (n + 255) / 256;
	kernel_selectValid<<<blk, 256, 0, stream>>>(d_in, n, d_out, d_numOut);
}

} // namespace MVS
