#include <iostream>
#include <device_launch_parameters.h>

#include "internal.h"
#include "limits.hpp"
#include "safe_call.hpp"
#include "kernel_containers.h"
#include "cutil_math.h"
#include "cuda_type.cuh"

namespace daniel_slam {

    using pcl::gpu::divUp;
    using pcl::gpu::PtrStepSz;
    using pcl::gpu::PtrStep;
    using pcl::device::numeric_limits;

    __global__ void
    computeVmapKernel(const PtrStepSz<unsigned short> depth, PtrStep<float> vmap, float fx_inv, float fy_inv, float cx,
                      float cy) {
        int u = threadIdx.x + blockIdx.x * blockDim.x;
        int v = threadIdx.y + blockIdx.y * blockDim.y;

        if (u < depth.cols && v < depth.rows) {
            float z = depth.ptr(v)[u] / 5000.f;

            if (z != 0) {
                float vx = z * (u - cx) * fx_inv;
                float vy = z * (v - cy) * fy_inv;
                float vz = z;

                vmap.ptr(v)[u] = vx;
                vmap.ptr(v + depth.rows)[u] = vy;
                vmap.ptr(v + depth.rows * 2)[u] = vz;
            } else
                vmap.ptr(v)[u] = numeric_limits<float>::quiet_NaN();

        }
    }

    __global__ void
    computeNmapKernel(int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap) {
        int u = threadIdx.x + blockIdx.x * blockDim.x;
        int v = threadIdx.y + blockIdx.y * blockDim.y;

        if (u >= cols || v >= rows)
            return;

        if (u == cols - 1 || v == rows - 1) {
            nmap.ptr(v)[u] = numeric_limits<float>::quiet_NaN();
            return;
        }

        float3 v00, v01, v10;
        v00.x = vmap.ptr(v)[u];
        v01.x = vmap.ptr(v)[u + 1];
        v10.x = vmap.ptr(v + 1)[u];

        if (!isnan(v00.x) && !isnan(v01.x) && !isnan(v10.x)) {
            v00.y = vmap.ptr(v + rows)[u];
            v01.y = vmap.ptr(v + rows)[u + 1];
            v10.y = vmap.ptr(v + 1 + rows)[u];

            v00.z = vmap.ptr(v + 2 * rows)[u];
            v01.z = vmap.ptr(v + 2 * rows)[u + 1];
            v10.z = vmap.ptr(v + 1 + 2 * rows)[u];

            float3 r = normalize(cross(v01 - v00, v10 - v00));

            nmap.ptr(v)[u] = r.x;
            nmap.ptr(v + rows)[u] = r.y;
            nmap.ptr(v + 2 * rows)[u] = r.z;
        } else
            nmap.ptr(v)[u] = numeric_limits<float>::quiet_NaN();
    }

    __global__ void
    pyrDownGaussKernel(const PtrStepSz<unsigned char> src, PtrStepSz<unsigned char> dst, float sigma_color) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= dst.cols || y >= dst.rows)
            return;

        const int D = 5;

        int center = src.ptr(2 * y)[2 * x];

        int x_mi = max(0, 2 * x - D / 2) - 2 * x;
        int y_mi = max(0, 2 * y - D / 2) - 2 * y;

        int x_ma = min(src.cols, 2 * x - D / 2 + D) - 2 * x;
        int y_ma = min(src.rows, 2 * y - D / 2 + D) - 2 * y;

        float sum = 0;
        float wall = 0;

        float weights[] = {0.375f, 0.25f, 0.0625f};

        for (int yi = y_mi; yi < y_ma; ++yi)
            for (int xi = x_mi; xi < x_ma; ++xi) {
                int val = src.ptr(2 * y + yi)[2 * x + xi];

                if (abs(val - center) < 3 * sigma_color) {
                    sum += val * weights[abs(xi)] * weights[abs(yi)];
                    wall += weights[abs(xi)] * weights[abs(yi)];
                }
            }

        dst.ptr(y)[x] = static_cast<int>(sum / wall);
    }

    inline int getGridDim(int x, int y) {
        return (x + y - 1) / y;
    }

    void pyrDown(const DepthDevice &src, DepthDevice &dst) {
        dst.create(src.rows() / 2, src.cols() / 2);

        dim3 block(32, 8);
        dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

        const float sigma_color = 30;

        pyrDownGaussKernel << < grid, block >> > (src, dst, sigma_color);
        cudaSafeCall (cudaGetLastError());
    };

    void createVMap(const CameraIntr &camera, const DepthDevice &depth, MapDevice &vmap) {
        vmap.create(depth.rows() * 3, depth.cols());

        dim3 block(32, 8);
        dim3 grid(1, 1, 1);
        grid.x = divUp(depth.cols(), block.x);
        grid.y = divUp(depth.rows(), block.y);

        float fx = camera.fx, cx = camera.cx;
        float fy = camera.fy, cy = camera.cy;

        computeVmapKernel << < grid, block >> > (depth, vmap, 1.f / fx, 1.f / fy, cx, cy);
        cudaSafeCall (cudaGetLastError());
    }

    void createNmap(const MapDevice &vmap, MapDevice &nmap) {
        nmap.create(vmap.rows(), vmap.cols());

        int rows = vmap.rows() / 3;
        int cols = vmap.cols();

        dim3 block(32, 8);
        dim3 grid(1, 1, 1);
        grid.x = divUp(cols, block.x);
        grid.y = divUp(rows, block.y);

        computeNmapKernel << < grid, block >> > (rows, cols, vmap, nmap);
        cudaSafeCall (cudaGetLastError());
    }

    __global__ void bgr2IntensityKernel(PtrStepSz<unsigned char> image, PtrStepSz<float> dst) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= dst.cols || y >= dst.rows)
            return;

        unsigned char r = image.ptr(y)[x * 3];
        unsigned char g = image.ptr(y)[x * 3 + 1];
        unsigned char b = image.ptr(y)[x * 3 + 2];

        float value = (float) r * 0.30f + (float) g * 0.59f + (float) b * 0.11f;

        dst.ptr(y)[x] = value / 256.0f;
    }

    void imageBGRToIntensity(DeviceArray2D<unsigned char> &image, DeviceArray2D<float> &dst) {
        dst.create(image.rows(), image.cols() / 3);

        dim3 block(32, 8);
        dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

        bgr2IntensityKernel << < grid, block >> > (image, dst);

        cudaSafeCall(cudaGetLastError());
    }

    __global__ void pyrDownKernelGaussF(const PtrStepSz<float> src, PtrStepSz<float> dst, float *gaussKernel) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= dst.cols || y >= dst.rows)
            return;

        const int D = 5;

        float center = src.ptr(2 * y)[2 * x];

        int tx = min(2 * x - D / 2 + D, src.cols - 1);
        int ty = min(2 * y - D / 2 + D, src.rows - 1);
        int cy = max(0, 2 * y - D / 2);

        float sum = 0;
        int count = 0;

        for (; cy < ty; ++cy) {
            for (int cx = max(0, 2 * x - D / 2); cx < tx; ++cx) {
                if (!isnan(src.ptr(cy)[cx])) {
                    sum += src.ptr(cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
                    count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
                }
            }
        }
        dst.ptr(y)[x] = (float) (sum / (float) count);
    }

    void pyrDownGaussF(const DeviceArray2D<float> &src, DeviceArray2D<float> &dst) {
        dst.create(src.rows() / 2, src.cols() / 2);

        dim3 block(32, 8);
        dim3 grid(getGridDim(dst.cols(), block.x), getGridDim(dst.rows(), block.y));

        const float gaussKernel[25] = {1, 4, 6, 4, 1,
                                       4, 16, 24, 16, 4,
                                       6, 24, 36, 24, 6,
                                       4, 16, 24, 16, 4,
                                       1, 4, 6, 4, 1};

        float *gauss_cuda;

        cudaMalloc((void **) &gauss_cuda, sizeof(float) * 25);
        cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);

        pyrDownKernelGaussF << < grid, block >> > (src, dst, gauss_cuda);
        cudaSafeCall (cudaGetLastError());

        cudaFree(gauss_cuda);
    };

    __inline__  __device__ JtJJtrSE3 warpReduceSum(JtJJtrSE3 val) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val.aa += __shfl_down(val.aa, offset);
            val.ab += __shfl_down(val.ab, offset);
            val.ac += __shfl_down(val.ac, offset);
            val.ad += __shfl_down(val.ad, offset);
            val.ae += __shfl_down(val.ae, offset);
            val.af += __shfl_down(val.af, offset);
            val.ag += __shfl_down(val.ag, offset);

            val.bb += __shfl_down(val.bb, offset);
            val.bc += __shfl_down(val.bc, offset);
            val.bd += __shfl_down(val.bd, offset);
            val.be += __shfl_down(val.be, offset);
            val.bf += __shfl_down(val.bf, offset);
            val.bg += __shfl_down(val.bg, offset);

            val.cc += __shfl_down(val.cc, offset);
            val.cd += __shfl_down(val.cd, offset);
            val.ce += __shfl_down(val.ce, offset);
            val.cf += __shfl_down(val.cf, offset);
            val.cg += __shfl_down(val.cg, offset);

            val.dd += __shfl_down(val.dd, offset);
            val.de += __shfl_down(val.de, offset);
            val.df += __shfl_down(val.df, offset);
            val.dg += __shfl_down(val.dg, offset);

            val.ee += __shfl_down(val.ee, offset);
            val.ef += __shfl_down(val.ef, offset);
            val.eg += __shfl_down(val.eg, offset);

            val.ff += __shfl_down(val.ff, offset);
            val.fg += __shfl_down(val.fg, offset);

            val.residual += __shfl_down(val.residual, offset);
            val.inliers += __shfl_down(val.inliers, offset);
        }

        return val;
    }

    __inline__  __device__ JtJJtrSE3 blockReduceSum(JtJJtrSE3 val) {
        static __shared__ JtJJtrSE3 shared[32];

        int lane = threadIdx.x % warpSize;

        int wid = threadIdx.x / warpSize;

        val = warpReduceSum(val);

        if (lane == 0) {
            shared[wid] = val;
        }
        __syncthreads();

        const JtJJtrSE3 zero = {0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0};

        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

        if (wid == 0) {
            val = warpReduceSum(val);
        }

        return val;
    }

    __global__ void reduceSum(JtJJtrSE3 *in, JtJJtrSE3 *out, int N) {
        JtJJtrSE3 sum = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0};

        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
            sum.add(in[i]);
        }

        sum = blockReduceSum(sum);

        if (threadIdx.x == 0) {
            out[blockIdx.x] = sum;
        }
    }

    struct ICPReduction {
        mat33 Rcurr;
        float3 tcurr;

        PtrStep<float> vmap_curr;
        PtrStep<float> nmap_curr;

        CameraIntr intr;

        PtrStep<float> vmap_prev;
        PtrStep<float> nmap_prev;

        float distThres;
        float angleThres;

        int cols;
        int rows;
        int N;

        JtJJtrSE3 *out;

        __device__ __forceinline__ bool
        search(int &x, int &y, float3 &n, float3 &d, float3 &s) {
            float3 vcurr;
            vcurr.x = vmap_curr.ptr(y)[x];
            vcurr.y = vmap_curr.ptr(y + rows)[x];
            vcurr.z = vmap_curr.ptr(y + 2 * rows)[x];

            float3 vcurr_cp = Rcurr * vcurr + tcurr;

            int2 ukr;
            ukr.x = __float2int_rn(vcurr_cp.x * intr.fx / vcurr_cp.z + intr.cx);
            ukr.y = __float2int_rn(vcurr_cp.y * intr.fy / vcurr_cp.z + intr.cy);

            if (ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || isnan(vcurr_cp.z))
                return false;

            float3 vprev;
            vprev.x = __ldg(&vmap_prev.ptr(ukr.y)[ukr.x]);
            vprev.y = __ldg(&vmap_prev.ptr(ukr.y + rows)[ukr.x]);
            vprev.z = __ldg(&vmap_prev.ptr(ukr.y + 2 * rows)[ukr.x]);

            float3 ncurr;
            ncurr.x = nmap_curr.ptr(y)[x];
            ncurr.y = nmap_curr.ptr(y + rows)[x];
            ncurr.z = nmap_curr.ptr(y + 2 * rows)[x];

            float3 ncurr_cp = Rcurr * ncurr;

            float3 nprev;
            nprev.x = __ldg(&nmap_prev.ptr(ukr.y)[ukr.x]);
            nprev.y = __ldg(&nmap_prev.ptr(ukr.y + rows)[ukr.x]);
            nprev.z = __ldg(&nmap_prev.ptr(ukr.y + 2 * rows)[ukr.x]);

            float dist = norm(vprev - vcurr_cp);
            float sine = norm(cross(ncurr_cp, nprev));

            n = nprev;
            d = vprev;
            s = vcurr_cp;

            return (sine < angleThres && dist <= distThres && !isnan(ncurr.x) && !isnan(nprev.x));
        }


        __device__ __forceinline__ JtJJtrSE3
        getProducts(int &i) {
            int y = i / cols;
            int x = i - (y * cols);

            float3 n_cp, d_cp, s_cp;

            bool found_coresp = search(x, y, n_cp, d_cp, s_cp);

            float row[7] = {0, 0, 0, 0, 0, 0, 0};

            if (found_coresp) {
                *(float3 *) &row[0] = n_cp;
                *(float3 *) &row[3] = cross(s_cp, n_cp);
                row[6] = dot(n_cp, d_cp - s_cp);
            }

            JtJJtrSE3 values = {row[0] * row[0],
                                row[0] * row[1],
                                row[0] * row[2],
                                row[0] * row[3],
                                row[0] * row[4],
                                row[0] * row[5],
                                row[0] * row[6],

                                row[1] * row[1],
                                row[1] * row[2],
                                row[1] * row[3],
                                row[1] * row[4],
                                row[1] * row[5],
                                row[1] * row[6],

                                row[2] * row[2],
                                row[2] * row[3],
                                row[2] * row[4],
                                row[2] * row[5],
                                row[2] * row[6],

                                row[3] * row[3],
                                row[3] * row[4],
                                row[3] * row[5],
                                row[3] * row[6],

                                row[4] * row[4],
                                row[4] * row[5],
                                row[4] * row[6],

                                row[5] * row[5],
                                row[5] * row[6],

                                row[6] * row[6],
                                found_coresp};

            return values;
        }

        __device__ __forceinline__ void
        operator()() {
            JtJJtrSE3 sum = {0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0};

            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
                JtJJtrSE3 val = getProducts(i);
                sum.add(val);
            }
            __syncthreads();

            sum = blockReduceSum(sum);

            __syncthreads();

            if (threadIdx.x == 0) {
                out[blockIdx.x] = sum;
            }
            __syncthreads();

        }
    };

    __global__ void icpKernel(ICPReduction icp) {
        icp();
    }

    void icpStep(const int layer,
                 const mat33 &Rcurr,
                 const float3 &tcurr,
                 const CameraIntr &intr,
                 const FrameNode &frame_pre,
                 const FrameNode &frame_cur,
                 float distThres,
                 float angleThres,
                 DeviceArray<JtJJtrSE3> &sum,
                 DeviceArray<JtJJtrSE3> &out,
                 float *matrixA_host,
                 float *vectorB_host,
                 float *residual_host) {
        int cols = frame_cur.cols >> layer;
        int rows = frame_cur.rows >> layer;

        ICPReduction icp;

        icp.Rcurr = Rcurr;
        icp.tcurr = tcurr;

        icp.vmap_curr = frame_cur.vertex_map[layer];
        icp.nmap_curr = frame_cur.normal_map[layer];

        icp.intr = intr;

        icp.vmap_prev = frame_pre.vertex_map[layer];
        icp.nmap_prev = frame_pre.normal_map[layer];

        icp.distThres = distThres;
        icp.angleThres = angleThres;

        icp.cols = cols;
        icp.rows = rows;

        icp.N = cols * rows;
        icp.out = sum;

        icpKernel << < 300, 1024 >> > (icp);

        reduceSum << < 1, 1024 >> > (sum, out, 300);

        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());

        float host_data[32];
        out.download((JtJJtrSE3 *) &host_data[0]);

        int shift = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 7; ++j) {
                float value = host_data[shift++];
                if (j == 6)
                    vectorB_host[i] = value;
                else
                    matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
            }
        }

        residual_host[0] = host_data[27];
        residual_host[1] = host_data[28];
    }

    __constant__ float gsobel_x3x3[9];
    __constant__ float gsobel_y3x3[9];

    __global__ void applyKernel(const PtrStepSz<float> src, PtrStep<float> dx, PtrStep<float> dy) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= src.cols || y >= src.rows)
            return;

        float dxVal = 0;
        float dyVal = 0;

        if ((y - 1 < 0) || (x - 1 < 0) || (y + 1 >= src.rows) || (x + 1 >= src.cols)) {
            dx.ptr(y)[x] = 0;
            dy.ptr(y)[x] = 0;
            return;
        }
        int kernelIndex = 0;
        for (int j = y - 1; j <= y + 1; j++) {
            for (int i = x - 1; i <= x + 1; i++) {
                dxVal += src.ptr(j)[i] * gsobel_x3x3[kernelIndex];
                dyVal += src.ptr(j)[i] * gsobel_y3x3[kernelIndex];
                kernelIndex++;
            }
        }

        dx.ptr(y)[x] = dxVal / 3.0f;
        dy.ptr(y)[x] = dyVal / 3.0f;
    }

    void
    computeDerivativeImages(DeviceArray2D<float> &src, DeviceArray2D<float> &dx, DeviceArray2D<float> &dy) {
        dx.create(src.rows(), src.cols());
        dy.create(src.rows(), src.cols());
        static bool once = false;

        if (!once) {
            float gsx3x3[9] = {-1.0f, 0.0, 1.0,
                               -1.0f, 0.0, 1.0,
                               -1.0f, 0.0, 1.0};

            float gsy3x3[9] = {-1.0f, -1.0f, -1.0f,
                               0.0, 0.0, 0.0,
                               1.0, 1.0, 1.0};

            cudaMemcpyToSymbol(gsobel_x3x3, gsx3x3, sizeof(float) * 9);
            cudaMemcpyToSymbol(gsobel_y3x3, gsy3x3, sizeof(float) * 9);

            cudaSafeCall(cudaGetLastError());
            cudaSafeCall(cudaDeviceSynchronize());

            once = true;
        }

        dim3 block(32, 8);
        dim3 grid(getGridDim(src.cols(), block.x), getGridDim(src.rows(), block.y));

        applyKernel << < grid, block >> > (src, dx, dy);

        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());
    }

    struct RGBReduction {
        mat33 Rcurr;
        float3 tcurr;
        PtrStepSz<float> vmap_curr;
        PtrStepSz<float> vmap_prev;
        PtrStepSz<float> image_cur;
        PtrStepSz<float> image_pre;
        PtrStepSz<float> dIdx;
        PtrStepSz<float> dIdy;
        CameraIntr camera;
        float minScale;
        float distThres;
        int cols;
        int rows;
        int N;

        JtJJtrSE3 *out;

        __device__ __forceinline__ bool
        search(int &x, int &y, float3 &v, float2 &dI, float &diff) const {
            float3 vcurr;
            vcurr.x = vmap_curr.ptr(y)[x];
            vcurr.y = vmap_curr.ptr(y + rows)[x];
            vcurr.z = vmap_curr.ptr(y + 2 * rows)[x];

            float3 vcurr_cp = Rcurr * vcurr + tcurr;

            int2 ukr;
            ukr.x = __float2int_rn(vcurr_cp.x * camera.fx / vcurr_cp.z + camera.cx);
            ukr.y = __float2int_rn(vcurr_cp.y * camera.fy / vcurr_cp.z + camera.cy);

            if (ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || isnan(vcurr_cp.z) )
                return false;

            float3 vprev;
            vprev.x = __ldg(&vmap_prev.ptr(ukr.y)[ukr.x]);
            vprev.y = __ldg(&vmap_prev.ptr(ukr.y + rows)[ukr.x]);
            vprev.z = __ldg(&vmap_prev.ptr(ukr.y + 2 * rows)[ukr.x]);

            float dist = norm(vprev - vcurr_cp);

            dI.x = dIdx.ptr(y)[x];
            dI.y = dIdy.ptr(y)[x];
            v = vcurr_cp;
            diff = image_cur.ptr(y)[x] - image_pre.ptr(ukr.y)[ukr.x];

            return (dI.x * dI.x + dI.y * dI.y < minScale && !isnan(diff) && !isnan(v.x) && dist <= distThres);

        }

        __device__ __forceinline__ JtJJtrSE3
        getProducts(int i) const {
            int y = i / cols;
            int x = i - (y * cols);

            float3 v;
            float2 dI;
            float diff;

            bool found_coresp = search(x, y, v, dI, diff);

            float row[7] = {0, 0, 0, 0, 0, 0, 0};

            if (found_coresp) {
                float invz = 1.0f / v.z;
                float v0 = dI.x * camera.fx * invz;
                float v1 = dI.y * camera.fy * invz;
                float v2 = -(v0 * v.x + v1 * v.y) * invz;
                row[0] = v0;
                row[1] = v1;
                row[2] = v2;
                row[3] = -v.z * v1 + v.y * v2;
                row[4] = v.z * v0 - v.x * v2;
                row[5] = -v.y * v0 + v.x * v1;
                row[6] = diff;
            } else {
                row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;
            }

            JtJJtrSE3 values = {row[0] * row[0],
                                row[0] * row[1],
                                row[0] * row[2],
                                row[0] * row[3],
                                row[0] * row[4],
                                row[0] * row[5],
                                row[0] * row[6],

                                row[1] * row[1],
                                row[1] * row[2],
                                row[1] * row[3],
                                row[1] * row[4],
                                row[1] * row[5],
                                row[1] * row[6],

                                row[2] * row[2],
                                row[2] * row[3],
                                row[2] * row[4],
                                row[2] * row[5],
                                row[2] * row[6],

                                row[3] * row[3],
                                row[3] * row[4],
                                row[3] * row[5],
                                row[3] * row[6],

                                row[4] * row[4],
                                row[4] * row[5],
                                row[4] * row[6],

                                row[5] * row[5],
                                row[5] * row[6],

                                row[6] * row[6],
                                found_coresp};
            return values;
        }

        __device__ __forceinline__ void
        operator()() const {
            JtJJtrSE3 sum = {0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0};

            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
                JtJJtrSE3 val = getProducts(i);

                sum.add(val);
            }
            __syncthreads();

            sum = blockReduceSum(sum);

            __syncthreads();

            if (threadIdx.x == 0) {
                out[blockIdx.x] = sum;
            }
        }
    };

    __global__ void rgbKernel(const RGBReduction rgb) {
        rgb();
    }

    void rgbStep(const int layer,
                 const mat33 &Rcurr,
                 const float3 &tcurr,
                 const CameraIntr &camera,
                 const FrameNode &frame_pre,
                 const FrameNode &frame_cur,
                 float minScale,
                 float distThres,
                 DeviceArray<JtJJtrSE3> &sum,
                 DeviceArray<JtJJtrSE3> &out,
                 float *matrixA_host,
                 float *vectorB_host,
                 float *residual_host) {

        RGBReduction rgb;

        rgb.minScale = minScale;
        rgb.distThres = distThres;
        rgb.Rcurr = Rcurr;
        rgb.tcurr = tcurr;
        rgb.cols = frame_cur.cols >> layer;
        rgb.rows = frame_cur.rows >> layer;
        rgb.vmap_curr = frame_cur.vertex_map[layer];
        rgb.vmap_prev = frame_pre.vertex_map[layer];
        rgb.image_cur = frame_cur.image_device[layer];
        rgb.image_pre = frame_pre.image_device[layer];
        rgb.camera = camera;
        rgb.dIdx = frame_cur.dIdx[layer];
        rgb.dIdy = frame_cur.dIdy[layer];
        rgb.N = rgb.cols * rgb.rows;
        rgb.out = sum;

        rgbKernel << < 300, 1024 >> > (rgb);

        reduceSum << < 1, 300 >> > (sum, out, 300);

        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());

        float host_data[32];
        out.download((JtJJtrSE3 *) &host_data[0]);

        int shift = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 7; ++j) {
                float value = host_data[shift++];
                if (j == 6)
                    vectorB_host[i] = value;
                else
                    matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
            }
        }
        residual_host[0] = host_data[27];
        residual_host[1] = host_data[28];
    }
}
