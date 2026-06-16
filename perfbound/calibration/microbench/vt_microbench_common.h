/*
 * Shared AscendC kernels for A.1 calibration microbenchmarks.
 *
 * These helpers intentionally use the current CANN AscendC surface
 * (`kernel_operator.h`) rather than the legacy CCE API.
 */
#ifndef VT_MICROBENCH_COMMON_H
#define VT_MICROBENCH_COMMON_H

#include "kernel_operator.h"

namespace VTCalib {

constexpr uint32_t kCubeM = 128;
constexpr uint32_t kCubeN = 128;
constexpr uint32_t kCubeKTile = 128;
constexpr uint32_t kCubeKTiles = 32;
constexpr uint32_t kCubeKTotal = kCubeKTile * kCubeKTiles;
constexpr uint32_t kCubeRepeat = 30;
constexpr uint32_t kCubeBlock = 16;
constexpr uint32_t kCubeBlockSize = kCubeBlock * kCubeBlock;

constexpr uint32_t kVectorElements = 256;
constexpr uint32_t kVectorRepeat = 10000;
constexpr uint32_t kMteElements = 131072;
constexpr uint32_t kMteRepeat = 2048;
constexpr uint32_t kMteWarmup = 768;
constexpr uint32_t kMteMeasured = 1280;
constexpr uint32_t kMteChunkElements = 32768;
constexpr uint32_t kMteL0Elements = 8192;
constexpr uint32_t kHbmAllCoreCores = 20;
constexpr uint32_t kHbmAllCoreChunksPerCore = 64;
constexpr uint32_t kHbmAllCoreElementsPerCore =
    kMteElements * kHbmAllCoreChunksPerCore;
constexpr uint32_t kHbmAllCoreTotalElements =
    kHbmAllCoreCores * kHbmAllCoreElementsPerCore;

__aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

class CubePeakFp16Kernel {
public:
    __aicore__ inline CubePeakFp16Kernel() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c)
    {
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), kCubeM * kCubeKTotal);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), kCubeKTotal * kCubeN);
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c), kCubeM * kCubeN);
        pipe.InitBuffer(a1Queue, 1, kCubeM * kCubeKTile * sizeof(half));
        pipe.InitBuffer(a2Queue, 1, kCubeM * kCubeKTile * sizeof(half));
        pipe.InitBuffer(b1Queue, 1, kCubeKTile * kCubeN * sizeof(half));
        pipe.InitBuffer(b2Queue, 1, kCubeKTile * kCubeN * sizeof(half));
        pipe.InitBuffer(c1Queue, 1, kCubeM * kCubeN * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        for (uint32_t repeat = 0; repeat < kCubeRepeat; ++repeat) {
            for (uint32_t kTile = 0; kTile < kCubeKTiles; ++kTile) {
                CopyIn(kTile);
                LoadA();
                LoadB();
                Compute();
                CopyOut();
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t kTile)
    {
        AscendC::LocalTensor<half> a1 = a1Queue.AllocTensor<half>();
        AscendC::LocalTensor<half> b1 = b1Queue.AllocTensor<half>();

        AscendC::Nd2NzParams aParams;
        aParams.ndNum = 1;
        aParams.nValue = kCubeM;
        aParams.dValue = kCubeKTile;
        aParams.srcNdMatrixStride = 0;
        aParams.srcDValue = kCubeKTotal;
        aParams.dstNzC0Stride = CeilDiv(kCubeM, kCubeBlock) * kCubeBlock;
        aParams.dstNzNStride = 1;
        aParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1, aGm[kTile * kCubeKTile], aParams);

        AscendC::Nd2NzParams bParams;
        bParams.ndNum = 1;
        bParams.nValue = kCubeKTile;
        bParams.dValue = kCubeN;
        bParams.srcNdMatrixStride = 0;
        bParams.srcDValue = kCubeN;
        bParams.dstNzC0Stride = CeilDiv(kCubeKTile, kCubeBlock) * kCubeBlock;
        bParams.dstNzNStride = 1;
        bParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1, bGm[kTile * kCubeKTile * kCubeN], bParams);

        a1Queue.EnQue(a1);
        b1Queue.EnQue(b1);
    }

    __aicore__ inline void LoadA()
    {
        AscendC::LocalTensor<half> a1 = a1Queue.DeQue<half>();
        AscendC::LocalTensor<half> a2 = a2Queue.AllocTensor<half>();
        AscendC::LoadData2DParams params;
        params.repeatTimes = CeilDiv(kCubeKTile, kCubeBlock);
        params.srcStride = CeilDiv(kCubeM, kCubeBlock);
        params.dstGap = 0;
        params.ifTranspose = false;
        for (uint32_t i = 0; i < CeilDiv(kCubeM, kCubeBlock); ++i) {
            AscendC::LoadData(a2[i * CeilDiv(kCubeKTile, kCubeBlock) * kCubeBlockSize],
                              a1[i * kCubeBlockSize], params);
        }
        a2Queue.EnQue(a2);
        a1Queue.FreeTensor(a1);
    }

    __aicore__ inline void LoadB()
    {
        AscendC::LocalTensor<half> b1 = b1Queue.DeQue<half>();
        AscendC::LocalTensor<half> b2 = b2Queue.AllocTensor<half>();
        AscendC::LoadData2DParams params;
        params.repeatTimes = CeilDiv(kCubeN, kCubeBlock);
        params.srcStride = CeilDiv(kCubeKTile, kCubeBlock);
        params.dstGap = 0;
        params.ifTranspose = true;
        for (uint32_t i = 0; i < CeilDiv(kCubeKTile, kCubeBlock); ++i) {
            AscendC::LoadData(b2[i * CeilDiv(kCubeN, kCubeBlock) * kCubeBlockSize],
                              b1[i * kCubeBlockSize], params);
        }
        b2Queue.EnQue(b2);
        b1Queue.FreeTensor(b1);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> a2 = a2Queue.DeQue<half>();
        AscendC::LocalTensor<half> b2 = b2Queue.DeQue<half>();
        AscendC::LocalTensor<float> c1 = c1Queue.AllocTensor<float>();
        AscendC::MmadParams params;
        params.m = kCubeM;
        params.n = kCubeN;
        params.k = kCubeKTile;
        AscendC::Mmad(c1, a2, b2, params);
        c1Queue.EnQue(c1);
        a2Queue.FreeTensor(a2);
        b2Queue.FreeTensor(b2);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> c1 = c1Queue.DeQue<float>();
        AscendC::FixpipeParamsV220 params;
        params.nSize = kCubeN;
        params.mSize = kCubeM;
        params.srcStride = kCubeM;
        params.dstStride = kCubeN;
        params.ndNum = 1;
        params.srcNdStride = 0;
        params.dstNdStride = 0;
        AscendC::Fixpipe(cGm, c1, params);
        c1Queue.FreeTensor(c1);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> a1Queue;
    AscendC::TQue<AscendC::TPosition::A2, 1> a2Queue;
    AscendC::TQue<AscendC::TPosition::B1, 1> b1Queue;
    AscendC::TQue<AscendC::TPosition::B2, 1> b2Queue;
    AscendC::TQue<AscendC::TPosition::CO1, 1> c1Queue;
    AscendC::GlobalTensor<half> aGm;
    AscendC::GlobalTensor<half> bGm;
    AscendC::GlobalTensor<float> cGm;
};

class VectorElemwiseKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), kVectorElements);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y), kVectorElements);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(z), kVectorElements);
        pipe.InitBuffer(xQueue, 1, kVectorElements * sizeof(half));
        pipe.InitBuffer(yQueue, 1, kVectorElements * sizeof(half));
        pipe.InitBuffer(zQueue, 1, kVectorElements * sizeof(half));
    }

    __aicore__ inline void ProcessAdd()
    {
        AscendC::LocalTensor<half> x = xQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> y = yQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> z = zQueue.AllocTensor<half>();
        AscendC::DataCopy(x, xGm, kVectorElements);
        AscendC::DataCopy(y, yGm, kVectorElements);
        for (uint32_t i = 0; i < kVectorRepeat; ++i) {
            AscendC::Add(z, x, y, kVectorElements);
        }
        AscendC::DataCopy(zGm, z, kVectorElements);
    }

    __aicore__ inline void ProcessMul()
    {
        AscendC::LocalTensor<half> x = xQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> y = yQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> z = zQueue.AllocTensor<half>();
        AscendC::DataCopy(x, xGm, kVectorElements);
        AscendC::DataCopy(y, yGm, kVectorElements);
        for (uint32_t i = 0; i < kVectorRepeat; ++i) {
            AscendC::Mul(z, x, y, kVectorElements);
        }
        AscendC::DataCopy(zGm, z, kVectorElements);
    }

    __aicore__ inline void ProcessMax()
    {
        AscendC::LocalTensor<half> x = xQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> y = yQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> z = zQueue.AllocTensor<half>();
        AscendC::DataCopy(x, xGm, kVectorElements);
        AscendC::DataCopy(y, yGm, kVectorElements);
        for (uint32_t i = 0; i < kVectorRepeat; ++i) {
            AscendC::Max(z, x, y, kVectorElements);
        }
        AscendC::DataCopy(zGm, z, kVectorElements);
    }

    __aicore__ inline void ProcessMin()
    {
        AscendC::LocalTensor<half> x = xQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> y = yQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> z = zQueue.AllocTensor<half>();
        AscendC::DataCopy(x, xGm, kVectorElements);
        AscendC::DataCopy(y, yGm, kVectorElements);
        for (uint32_t i = 0; i < kVectorRepeat; ++i) {
            AscendC::Min(z, x, y, kVectorElements);
        }
        AscendC::DataCopy(zGm, z, kVectorElements);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> yQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> zQueue;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
};

// Scalar pipe (PIPE_S) sustained throughput.
//
// Measures the *scalar* ALU, not the vector SIMD engine.  A dependent
// fused-multiply-add chain on a single scalar register prevents the
// AscendC compiler from auto-vectorising the loop (each iteration depends
// on the previous accumulator), so the work is forced onto the scalar
// issue path.  Two FLOP per iteration (one multiply, one add).
//
// The accumulator is written to GM at the end so dead-code elimination
// cannot drop the loop.  N_iter = kScalarRepeat dependent FMAs.
constexpr uint32_t kScalarRepeat = 1000000;

class ScalarPeakKernel {
public:
    __aicore__ inline void Init(GM_ADDR z)
    {
        // 8 floats = one 32-byte block (the minimum UB<->GM DataCopy granule).
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(z), 8);
        pipe.InitBuffer(zQueue, 1, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        // Dependent scalar FMA chain: acc = acc * c1 + c2.  Each iteration
        // depends on the previous accumulator, so the compiler cannot
        // auto-vectorise and the work stays on the scalar issue path.
        // The exact constants are irrelevant to timing; they only keep the
        // recurrence from overflowing or collapsing to a constant.
        float acc = 1.0f;
        const float c1 = 1.0000001f;
        const float c2 = 0.0000001f;
        for (uint32_t i = 0; i < kScalarRepeat; ++i) {
            acc = acc * c1 + c2;
        }
        // Write the final accumulator to GM so dead-code elimination cannot
        // drop the loop.  Broadcast into a 32-byte UB block, then copy out.
        AscendC::LocalTensor<float> out = zQueue.AllocTensor<float>();
        for (uint32_t i = 0; i < 8; ++i) {
            out.SetValue(i, acc);
        }
        zQueue.EnQue(out);
        AscendC::LocalTensor<float> deq = zQueue.DeQue<float>();
        AscendC::DataCopy(zGm, deq, 8);
        zQueue.FreeTensor(deq);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> zQueue;
    AscendC::GlobalTensor<float> zGm;
};

class VectorTransKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z)
    {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), kVectorElements);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(z), kVectorElements);
        pipe.InitBuffer(xQueue, 1, kVectorElements * sizeof(half));
        pipe.InitBuffer(zQueue, 1, kVectorElements * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<half> x = xQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> z = zQueue.AllocTensor<half>();
        AscendC::DataCopy(x, xGm, kVectorElements);
        for (uint32_t i = 0; i < kVectorRepeat; ++i) {
            AscendC::Exp(z, x, kVectorElements);
            AscendC::Ln(z, z, kVectorElements);
            AscendC::Sqrt(z, z, kVectorElements);
            AscendC::Rsqrt(z, z, kVectorElements);
        }
        AscendC::DataCopy(zGm, z, kVectorElements);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> zQueue;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> zGm;
};

class MteGmUbKernel {
public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, uint32_t startIter, uint32_t iterCount)
    {
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(src), kMteElements * kMteRepeat);
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dst), kMteElements * kMteRepeat);
        pipe.InitBuffer(queue, 1, kMteChunkElements * sizeof(half));
        start = startIter;
        count = iterCount;
    }

    __aicore__ inline void GmToUb()
    {
        AscendC::LocalTensor<half> local = queue.AllocTensor<half>();
        for (uint32_t iter = 0; iter < count; ++iter) {
            uint32_t i = start + iter;
            for (uint32_t chunk = 0; chunk < kMteElements / kMteChunkElements; ++chunk) {
                uint32_t offset = chunk * kMteChunkElements;
                AscendC::DataCopy(local, srcGm[i * kMteElements + offset], kMteChunkElements);
                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::DataCopy(dstGm[i * kMteElements + offset], local, kMteChunkElements);
                AscendC::PipeBarrier<PIPE_MTE3>();
            }
        }
    }

    __aicore__ inline void UbToGm()
    {
        AscendC::LocalTensor<half> local = queue.AllocTensor<half>();
        for (uint32_t iter = 0; iter < count; ++iter) {
            uint32_t i = start + iter;
            for (uint32_t chunk = 0; chunk < kMteElements / kMteChunkElements; ++chunk) {
                uint32_t offset = chunk * kMteChunkElements;
                AscendC::DataCopy(local, srcGm[i * kMteElements + offset], kMteChunkElements);
                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::DataCopy(dstGm[i * kMteElements + offset], local, kMteChunkElements);
                AscendC::PipeBarrier<PIPE_MTE3>();
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> queue;
    AscendC::GlobalTensor<half> srcGm;
    AscendC::GlobalTensor<half> dstGm;
    uint32_t start = 0;
    uint32_t count = kMteRepeat;
};

class MteCubePathKernel {
public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, uint32_t startIter, uint32_t iterCount)
    {
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(src), kMteElements * kMteRepeat);
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dst), kMteElements);
        pipe.InitBuffer(a1Queue, 1, kMteElements * sizeof(half));
        pipe.InitBuffer(a2Queue, 1, kMteL0Elements * sizeof(half));
        start = startIter;
        count = iterCount;
    }

    __aicore__ inline void GmToL1()
    {
        AscendC::LocalTensor<half> local = a1Queue.AllocTensor<half>();
        for (uint32_t iter = 0; iter < count; ++iter) {
            uint32_t i = start + iter;
            AscendC::DataCopy(local, srcGm[i * kMteElements], kMteElements);
        }
    }

    __aicore__ inline void L1ToL0A()
    {
        AscendC::LocalTensor<half> a1 = a1Queue.AllocTensor<half>();
        AscendC::LocalTensor<half> a2 = a2Queue.AllocTensor<half>();
        AscendC::DataCopy(a1, srcGm[start * kMteElements], kMteElements);
        AscendC::LoadData2DParams params;
        params.repeatTimes = CeilDiv(kMteL0Elements, kCubeBlockSize);
        params.srcStride = 1;
        params.dstGap = 0;
        params.ifTranspose = false;
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t chunk = 0; chunk < kMteElements / kMteL0Elements; ++chunk) {
                AscendC::LoadData(a2, a1[chunk * kMteL0Elements], params);
            }
        }
        AscendC::DataCopy(dstGm, a1, kMteElements);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> a1Queue;
    AscendC::TQue<AscendC::TPosition::A2, 1> a2Queue;
    AscendC::GlobalTensor<half> srcGm;
    AscendC::GlobalTensor<half> dstGm;
    uint32_t start = 0;
    uint32_t count = kMteRepeat;
};

class MandatoryHandoffKernel {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t K)
    {
        handoffK = K;
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), kCubeM * kCubeKTotal);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), kCubeKTotal * kCubeN);
        cFloatGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c), kCubeM * kCubeN);
        cHalfGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c), kVectorElements);
        if ASCEND_IS_AIC {
            pipe.InitBuffer(a1Queue, 1, kCubeM * kCubeKTile * sizeof(half));
            pipe.InitBuffer(a2Queue, 1, kCubeM * kCubeKTile * sizeof(half));
            pipe.InitBuffer(b1Queue, 1, kCubeKTile * kCubeN * sizeof(half));
            pipe.InitBuffer(b2Queue, 1, kCubeKTile * kCubeN * sizeof(half));
            pipe.InitBuffer(c1Queue, 1, kCubeM * kCubeN * sizeof(float));
        }
        if ASCEND_IS_AIV {
            pipe.InitBuffer(xQueue, 1, kVectorElements * sizeof(half));
            pipe.InitBuffer(yQueue, 1, kVectorElements * sizeof(half));
            pipe.InitBuffer(zQueue, 1, kVectorElements * sizeof(half));
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            ProcessCube();
            AscendC::SyncAll<false>();
        }
        if ASCEND_IS_AIV {
            AscendC::SyncAll<false>();
            ProcessVector();
        }
    }

private:
    __aicore__ inline void ProcessCube()
    {
        uint32_t kTiles = CeilDiv(handoffK, kCubeKTile);
        if (kTiles == 0) {
            kTiles = 1;
        }
        if (kTiles > kCubeKTiles) {
            kTiles = kCubeKTiles;
        }
        for (uint32_t kTile = 0; kTile < kTiles; ++kTile) {
            CopyIn(kTile);
            LoadA();
            LoadB();
            Compute();
            CopyOut();
        }
    }

    __aicore__ inline void CopyIn(uint32_t kTile)
    {
        AscendC::LocalTensor<half> a1 = a1Queue.AllocTensor<half>();
        AscendC::LocalTensor<half> b1 = b1Queue.AllocTensor<half>();

        AscendC::Nd2NzParams aParams;
        aParams.ndNum = 1;
        aParams.nValue = kCubeM;
        aParams.dValue = kCubeKTile;
        aParams.srcNdMatrixStride = 0;
        aParams.srcDValue = kCubeKTotal;
        aParams.dstNzC0Stride = CeilDiv(kCubeM, kCubeBlock) * kCubeBlock;
        aParams.dstNzNStride = 1;
        aParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1, aGm[kTile * kCubeKTile], aParams);

        AscendC::Nd2NzParams bParams;
        bParams.ndNum = 1;
        bParams.nValue = kCubeKTile;
        bParams.dValue = kCubeN;
        bParams.srcNdMatrixStride = 0;
        bParams.srcDValue = kCubeN;
        bParams.dstNzC0Stride = CeilDiv(kCubeKTile, kCubeBlock) * kCubeBlock;
        bParams.dstNzNStride = 1;
        bParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1, bGm[kTile * kCubeKTile * kCubeN], bParams);

        a1Queue.EnQue(a1);
        b1Queue.EnQue(b1);
    }

    __aicore__ inline void LoadA()
    {
        AscendC::LocalTensor<half> a1 = a1Queue.DeQue<half>();
        AscendC::LocalTensor<half> a2 = a2Queue.AllocTensor<half>();
        AscendC::LoadData2DParams params;
        params.repeatTimes = CeilDiv(kCubeKTile, kCubeBlock);
        params.srcStride = CeilDiv(kCubeM, kCubeBlock);
        params.dstGap = 0;
        params.ifTranspose = false;
        for (uint32_t i = 0; i < CeilDiv(kCubeM, kCubeBlock); ++i) {
            AscendC::LoadData(a2[i * CeilDiv(kCubeKTile, kCubeBlock) * kCubeBlockSize],
                              a1[i * kCubeBlockSize], params);
        }
        a2Queue.EnQue(a2);
        a1Queue.FreeTensor(a1);
    }

    __aicore__ inline void LoadB()
    {
        AscendC::LocalTensor<half> b1 = b1Queue.DeQue<half>();
        AscendC::LocalTensor<half> b2 = b2Queue.AllocTensor<half>();
        AscendC::LoadData2DParams params;
        params.repeatTimes = CeilDiv(kCubeN, kCubeBlock);
        params.srcStride = CeilDiv(kCubeKTile, kCubeBlock);
        params.dstGap = 0;
        params.ifTranspose = true;
        for (uint32_t i = 0; i < CeilDiv(kCubeKTile, kCubeBlock); ++i) {
            AscendC::LoadData(b2[i * CeilDiv(kCubeN, kCubeBlock) * kCubeBlockSize],
                              b1[i * kCubeBlockSize], params);
        }
        b2Queue.EnQue(b2);
        b1Queue.FreeTensor(b1);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> a2 = a2Queue.DeQue<half>();
        AscendC::LocalTensor<half> b2 = b2Queue.DeQue<half>();
        AscendC::LocalTensor<float> c1 = c1Queue.AllocTensor<float>();
        AscendC::MmadParams params;
        params.m = kCubeM;
        params.n = kCubeN;
        params.k = kCubeKTile;
        AscendC::Mmad(c1, a2, b2, params);
        c1Queue.EnQue(c1);
        a2Queue.FreeTensor(a2);
        b2Queue.FreeTensor(b2);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> c1 = c1Queue.DeQue<float>();
        AscendC::FixpipeParamsV220 params;
        params.nSize = kCubeN;
        params.mSize = kCubeM;
        params.srcStride = kCubeM;
        params.dstStride = kCubeN;
        params.ndNum = 1;
        params.srcNdStride = 0;
        params.dstNdStride = 0;
        AscendC::Fixpipe(cFloatGm, c1, params);
        c1Queue.FreeTensor(c1);
    }

    __aicore__ inline void ProcessVector()
    {
        AscendC::LocalTensor<half> x = xQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> y = yQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> z = zQueue.AllocTensor<half>();
        AscendC::DataCopy(x, cHalfGm, kVectorElements);
        AscendC::DataCopy(y, cHalfGm, kVectorElements);
        AscendC::Add(z, x, y, kVectorElements);
        AscendC::DataCopy(cHalfGm, z, kVectorElements);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> a1Queue;
    AscendC::TQue<AscendC::TPosition::A2, 1> a2Queue;
    AscendC::TQue<AscendC::TPosition::B1, 1> b1Queue;
    AscendC::TQue<AscendC::TPosition::B2, 1> b2Queue;
    AscendC::TQue<AscendC::TPosition::CO1, 1> c1Queue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> yQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> zQueue;
    AscendC::GlobalTensor<half> aGm;
    AscendC::GlobalTensor<half> bGm;
    AscendC::GlobalTensor<float> cFloatGm;
    AscendC::GlobalTensor<half> cHalfGm;
    uint32_t handoffK = kCubeKTile;
};

// FixPipe L0C→GM sustained bandwidth microbenchmark.
//
// Measures the sustained FixPipe (L0C→GM) transfer rate by performing
// repeated MMAD→Fixpipe cycles on the Cube engine.  The bandwidth is:
//   BW_l0c_to_gm = (M * N * sizeof(float) * N_fixpipe_iters) / time_us
//
// Uses the same Cube pipeline as CubePeakFp16Kernel but isolates the
// Fixpipe (CopyOut) stage by running many fixpipe-only iterations after
// a single MMAD produces data in L0C.
class MteFixpipeKernel {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                uint32_t startIter, uint32_t iterCount)
    {
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), kCubeM * kCubeKTotal);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), kCubeKTotal * kCubeN);
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c), kCubeM * kCubeN);
        pipe.InitBuffer(a1Queue, 1, kCubeM * kCubeKTile * sizeof(half));
        pipe.InitBuffer(a2Queue, 1, kCubeM * kCubeKTile * sizeof(half));
        pipe.InitBuffer(b1Queue, 1, kCubeKTile * kCubeN * sizeof(half));
        pipe.InitBuffer(b2Queue, 1, kCubeKTile * kCubeN * sizeof(half));
        pipe.InitBuffer(c1Queue, 1, kCubeM * kCubeN * sizeof(float));
        start = startIter;
        count = iterCount;
    }

    __aicore__ inline void Process()
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }

        // Initialize one valid L0C tile, then retain it for the FixPipe loop.
        CopyIn(0);
        LoadA();
        LoadB();
        AscendC::LocalTensor<half> a2 = a2Queue.DeQue<half>();
        AscendC::LocalTensor<half> b2 = b2Queue.DeQue<half>();
        AscendC::LocalTensor<float> c1 = c1Queue.AllocTensor<float>();
        AscendC::MmadParams mmadParams;
        mmadParams.m = kCubeM;
        mmadParams.n = kCubeN;
        mmadParams.k = kCubeKTile;
        AscendC::Mmad(c1, a2, b2, mmadParams);
        c1Queue.EnQue(c1);
        c1 = c1Queue.DeQue<float>();
        a2Queue.FreeTensor(a2);
        b2Queue.FreeTensor(b2);

        AscendC::FixpipeParamsV220 params;
        params.nSize = kCubeN;
        params.mSize = kCubeM;
        params.srcStride = kCubeM;
        params.dstStride = kCubeN;
        params.ndNum = 1;
        params.srcNdStride = 0;
        params.dstNdStride = 0;
        for (uint32_t iter = 0; iter < count; ++iter) {
            AscendC::Fixpipe(cGm, c1, params);
        }
        c1Queue.FreeTensor(c1);
    }

private:
    __aicore__ inline void CopyIn(uint32_t kTile)
    {
        AscendC::LocalTensor<half> a1 = a1Queue.AllocTensor<half>();
        AscendC::LocalTensor<half> b1 = b1Queue.AllocTensor<half>();

        AscendC::Nd2NzParams aParams;
        aParams.ndNum = 1;
        aParams.nValue = kCubeM;
        aParams.dValue = kCubeKTile;
        aParams.srcNdMatrixStride = 0;
        aParams.srcDValue = kCubeKTotal;
        aParams.dstNzC0Stride = CeilDiv(kCubeM, kCubeBlock) * kCubeBlock;
        aParams.dstNzNStride = 1;
        aParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1, aGm[kTile * kCubeKTile], aParams);

        AscendC::Nd2NzParams bParams;
        bParams.ndNum = 1;
        bParams.nValue = kCubeKTile;
        bParams.dValue = kCubeN;
        bParams.srcNdMatrixStride = 0;
        bParams.srcDValue = kCubeN;
        bParams.dstNzC0Stride = CeilDiv(kCubeKTile, kCubeBlock) * kCubeBlock;
        bParams.dstNzNStride = 1;
        bParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1, bGm[kTile * kCubeKTile * kCubeN], bParams);

        a1Queue.EnQue(a1);
        b1Queue.EnQue(b1);
    }

    __aicore__ inline void LoadA()
    {
        AscendC::LocalTensor<half> a1 = a1Queue.DeQue<half>();
        AscendC::LocalTensor<half> a2 = a2Queue.AllocTensor<half>();
        AscendC::LoadData2DParams params;
        params.repeatTimes = CeilDiv(kCubeKTile, kCubeBlock);
        params.srcStride = CeilDiv(kCubeM, kCubeBlock);
        params.dstGap = 0;
        params.ifTranspose = false;
        for (uint32_t i = 0; i < CeilDiv(kCubeM, kCubeBlock); ++i) {
            AscendC::LoadData(a2[i * CeilDiv(kCubeKTile, kCubeBlock) * kCubeBlockSize],
                              a1[i * kCubeBlockSize], params);
        }
        a2Queue.EnQue(a2);
        a1Queue.FreeTensor(a1);
    }

    __aicore__ inline void LoadB()
    {
        AscendC::LocalTensor<half> b1 = b1Queue.DeQue<half>();
        AscendC::LocalTensor<half> b2 = b2Queue.AllocTensor<half>();
        AscendC::LoadData2DParams params;
        params.repeatTimes = CeilDiv(kCubeN, kCubeBlock);
        params.srcStride = CeilDiv(kCubeKTile, kCubeBlock);
        params.dstGap = 0;
        params.ifTranspose = true;
        for (uint32_t i = 0; i < CeilDiv(kCubeKTile, kCubeBlock); ++i) {
            AscendC::LoadData(b2[i * CeilDiv(kCubeN, kCubeBlock) * kCubeBlockSize],
                              b1[i * kCubeBlockSize], params);
        }
        b2Queue.EnQue(b2);
        b1Queue.FreeTensor(b1);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> a1Queue;
    AscendC::TQue<AscendC::TPosition::A2, 1> a2Queue;
    AscendC::TQue<AscendC::TPosition::B1, 1> b1Queue;
    AscendC::TQue<AscendC::TPosition::B2, 1> b2Queue;
    AscendC::TQue<AscendC::TPosition::CO1, 1> c1Queue;
    AscendC::GlobalTensor<half> aGm;
    AscendC::GlobalTensor<half> bGm;
    AscendC::GlobalTensor<float> cGm;
    uint32_t start = 0;
    uint32_t count = kMteMeasured;
};

// All-core HBM sustained bandwidth microbenchmark.
//
// Launches on ALL 20 AIC cores simultaneously, each performing sustained
// GM→L1 DataCopy reads.  Measures the aggregate HBM bandwidth when all
// cores contend for memory.  This gives BW_hbm_allcore_sustained, which
// is the realistic per-core bandwidth under full load:
//   BW_hbm_allcore_sustained = (total_bytes_all_cores) / (max_core_time_us)
//
// Each core cycles through a disjoint 16 MiB region.  The aggregate 320 MiB
// footprint exceeds L2 capacity, so the measured traffic reaches HBM.  The
// per-core sustained rate is:
//   per_core_bw = (transfer_bytes * N_iter) / core_time_us
class MteAllCoreHbmKernel {
public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst,
                                uint32_t startIter, uint32_t iterCount)
    {
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(src), kHbmAllCoreTotalElements);
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dst), kMteElements);
        pipe.InitBuffer(a1Queue, 1, kMteElements * sizeof(half));
        start = startIter;
        count = iterCount;
    }

    __aicore__ inline void Process()
    {
        // ALL cores participate (no blockIdx guard)
        AscendC::LocalTensor<half> local = a1Queue.AllocTensor<half>();
        uint32_t coreBase =
            AscendC::GetBlockIdx() * kHbmAllCoreElementsPerCore;
        for (uint32_t iter = 0; iter < count; ++iter) {
            uint32_t chunk =
                (start + iter) % kHbmAllCoreChunksPerCore;
            AscendC::DataCopy(
                local,
                srcGm[coreBase + chunk * kMteElements],
                kMteElements);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> a1Queue;
    AscendC::GlobalTensor<half> srcGm;
    AscendC::GlobalTensor<half> dstGm;
    uint32_t start = 0;
    uint32_t count = kMteMeasured;
};

} // namespace VTCalib

#endif // VT_MICROBENCH_COMMON_H
