#include <cstdlib>
#include <iostream>
#include <string>

#include "acl/acl.h"
#include "aclrtlaunch_cube_peak_bf16.h"
#include "aclrtlaunch_cube_peak_fp16.h"
#include "aclrtlaunch_cube_peak_int8.h"
#include "aclrtlaunch_mandatory_handoff.h"
#include "aclrtlaunch_mte_gm_to_l1.h"
#include "aclrtlaunch_mte_gm_to_ub.h"
#include "aclrtlaunch_mte_hbm_allcore.h"
#include "aclrtlaunch_mte_l0c_to_gm.h"
#include "aclrtlaunch_mte_l1_to_l0a.h"
#include "aclrtlaunch_mte_ub_to_gm.h"
#include "aclrtlaunch_scalar_peak.h"
#include "aclrtlaunch_vector_peak_elemwise_add.h"
#include "aclrtlaunch_vector_peak_elemwise_max.h"
#include "aclrtlaunch_vector_peak_elemwise_min.h"
#include "aclrtlaunch_vector_peak_elemwise_mul.h"
#include "aclrtlaunch_vector_peak_transcendental.h"

namespace {

constexpr size_t kBufferBytes = 512 * 1024 * 1024;
constexpr uint32_t kCubeBlockDim = 1;
constexpr uint32_t kVectorBlockDim = 1;

void CheckAcl(aclError err, const char *expr, int line)
{
    if (err != ACL_ERROR_NONE) {
        std::cerr << "ACL failure at line " << line << ": " << expr << " -> " << err << std::endl;
        std::exit(2);
    }
}

#define CHECK_ACL(expr) CheckAcl((expr), #expr, __LINE__)

struct DeviceBuffers {
    void *a = nullptr;
    void *b = nullptr;
    void *c = nullptr;
};

void AllocateBuffers(DeviceBuffers &buffers)
{
    CHECK_ACL(aclrtMalloc(&buffers.a, kBufferBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&buffers.b, kBufferBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&buffers.c, kBufferBytes, ACL_MEM_MALLOC_HUGE_FIRST));
}

void FreeBuffers(DeviceBuffers &buffers)
{
    if (buffers.a != nullptr) {
        CHECK_ACL(aclrtFree(buffers.a));
    }
    if (buffers.b != nullptr) {
        CHECK_ACL(aclrtFree(buffers.b));
    }
    if (buffers.c != nullptr) {
        CHECK_ACL(aclrtFree(buffers.c));
    }
}

bool LaunchKernel(
    const std::string &kernel,
    aclrtStream stream,
    DeviceBuffers &buffers,
    uint32_t kValue,
    uint32_t mteStart,
    uint32_t mteIters,
    uint32_t blockDim)
{
    if (kernel == "cube_peak_fp16") {
        ACLRT_LAUNCH_KERNEL(cube_peak_fp16)(kCubeBlockDim, stream, buffers.a, buffers.b, buffers.c);
    } else if (kernel == "cube_peak_int8") {
        ACLRT_LAUNCH_KERNEL(cube_peak_int8)(kCubeBlockDim, stream, buffers.a, buffers.b, buffers.c);
    } else if (kernel == "cube_peak_bf16") {
        ACLRT_LAUNCH_KERNEL(cube_peak_bf16)(kCubeBlockDim, stream, buffers.a, buffers.b, buffers.c);
    } else if (kernel == "vector_peak_elemwise_add") {
        ACLRT_LAUNCH_KERNEL(vector_peak_elemwise_add)(kVectorBlockDim, stream, buffers.a, buffers.b, buffers.c);
    } else if (kernel == "vector_peak_elemwise_mul") {
        ACLRT_LAUNCH_KERNEL(vector_peak_elemwise_mul)(kVectorBlockDim, stream, buffers.a, buffers.b, buffers.c);
    } else if (kernel == "vector_peak_elemwise_max") {
        ACLRT_LAUNCH_KERNEL(vector_peak_elemwise_max)(kVectorBlockDim, stream, buffers.a, buffers.b, buffers.c);
    } else if (kernel == "vector_peak_elemwise_min") {
        ACLRT_LAUNCH_KERNEL(vector_peak_elemwise_min)(kVectorBlockDim, stream, buffers.a, buffers.b, buffers.c);
    } else if (kernel == "vector_peak_transcendental") {
        ACLRT_LAUNCH_KERNEL(vector_peak_transcendental)(kVectorBlockDim, stream, buffers.a, buffers.c);
    } else if (kernel == "scalar_peak") {
        ACLRT_LAUNCH_KERNEL(scalar_peak)(kVectorBlockDim, stream, buffers.c);
    } else if (kernel == "mte_gm_to_ub") {
        ACLRT_LAUNCH_KERNEL(mte_gm_to_ub)(kVectorBlockDim, stream, buffers.a, buffers.c, mteStart, mteIters);
    } else if (kernel == "mte_ub_to_gm") {
        ACLRT_LAUNCH_KERNEL(mte_ub_to_gm)(kVectorBlockDim, stream, buffers.a, buffers.c, mteStart, mteIters);
    } else if (kernel == "mte_gm_to_l1") {
        ACLRT_LAUNCH_KERNEL(mte_gm_to_l1)(kCubeBlockDim, stream, buffers.a, buffers.c, mteStart, mteIters);
    } else if (kernel == "mte_l1_to_l0a") {
        ACLRT_LAUNCH_KERNEL(mte_l1_to_l0a)(kCubeBlockDim, stream, buffers.a, buffers.c, mteStart, mteIters);
    } else if (kernel == "mte_l0c_to_gm") {
        // FixPipe (L0C->GM) sustained bandwidth; single Cube core (core 0 only).
        ACLRT_LAUNCH_KERNEL(mte_l0c_to_gm)(kCubeBlockDim, stream, buffers.a, buffers.b, buffers.c, mteStart, mteIters);
    } else if (kernel == "mte_hbm_allcore") {
        // All-core HBM: launch on all 20 AIC cores via --block-dim to contend for HBM.
        ACLRT_LAUNCH_KERNEL(mte_hbm_allcore)(blockDim, stream, buffers.a, buffers.c, mteStart, mteIters);
    } else if (kernel == "mandatory_handoff") {
        ACLRT_LAUNCH_KERNEL(mandatory_handoff)(kCubeBlockDim, stream, buffers.a, buffers.b, buffers.c, kValue);
    } else {
        return false;
    }
    return true;
}

std::string ArgValue(int argc, char **argv, const std::string &name, const std::string &fallback)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

} // namespace

int main(int argc, char **argv)
{
    const std::string kernel = ArgValue(argc, argv, "--kernel", "cube_peak_fp16");
    const int repeat = std::stoi(ArgValue(argc, argv, "--repeat", "30"));
    const uint32_t kValue = static_cast<uint32_t>(std::stoul(ArgValue(argc, argv, "--k", "128")));
    const uint32_t mteStart = static_cast<uint32_t>(std::stoul(ArgValue(argc, argv, "--mte-start", "0")));
    const uint32_t mteIters = static_cast<uint32_t>(std::stoul(ArgValue(argc, argv, "--mte-iters", "2048")));
    const uint32_t blockDim = static_cast<uint32_t>(std::stoul(ArgValue(argc, argv, "--block-dim", "1")));

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    DeviceBuffers buffers;
    AllocateBuffers(buffers);

    for (int i = 0; i < repeat; ++i) {
        if (!LaunchKernel(kernel, stream, buffers, kValue, mteStart, mteIters, blockDim)) {
            std::cerr << "Unknown kernel: " << kernel << std::endl;
            FreeBuffers(buffers);
            CHECK_ACL(aclrtDestroyStream(stream));
            CHECK_ACL(aclrtResetDevice(0));
            CHECK_ACL(aclFinalize());
            return 1;
        }
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    FreeBuffers(buffers);
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return 0;
}
