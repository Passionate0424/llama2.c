#include "runtime_hw_adapter.h"
#include "runtime_memory_map.h"

#include <stdio.h>
#include <string.h>

// 这层的职责不是“实现硬件算子”，而是把语义级 backend 调用翻译成
// SoC 实际能接受的 MMIO/DMA 作业提交流程。
// 第一版先把接口、地址口径和共享区边界冻结，真实寄存器协议后续替换进来。

#if defined(__GNUC__)
#define RUNTIME_SEC(section_name) __attribute__((section(section_name), aligned(64)))
#else
#define RUNTIME_SEC(section_name)
#endif

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define RUNTIME_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
#define RUNTIME_STATIC_ASSERT(cond, msg)
#endif

RUNTIME_STATIC_ASSERT(sizeof(RuntimeHwCmdEntry) == RUNTIME_HW_CMD_ENTRY_SIZE, "RuntimeHwCmdEntry must stay 32B");
RUNTIME_STATIC_ASSERT(sizeof(RuntimeHwCmpEntry) == RUNTIME_HW_CMP_ENTRY_SIZE, "RuntimeHwCmpEntry must stay 16B");
RUNTIME_STATIC_ASSERT((RUNTIME_ACCEL_CMDQ_SIZE % RUNTIME_HW_CMD_ENTRY_SIZE) == 0, "CMDQ size must align to cmd entry size");
RUNTIME_STATIC_ASSERT((RUNTIME_ACCEL_CMPQ_SIZE % RUNTIME_HW_CMP_ENTRY_SIZE) == 0, "CMPQ size must align to cmp entry size");

// 共享区 backing storage：
// - 在 host/HW_STUB 上直接作为可访问内存；
// - 在 SoC 上可替换成实际 uncached alias 映射，接口层不变。
static unsigned char g_accel_io_in[RUNTIME_ACCEL_IO_IN_SIZE] RUNTIME_SEC(".accel_shared");
static unsigned char g_accel_io_out[RUNTIME_ACCEL_IO_OUT_SIZE] RUNTIME_SEC(".accel_shared");
static unsigned char g_accel_param[RUNTIME_ACCEL_PARAM_SIZE] RUNTIME_SEC(".accel_shared");
static unsigned char g_key_cache_main_data[RUNTIME_KEY_CACHE_MAIN_DATA_SIZE] RUNTIME_SEC(".kv_main");
static unsigned char g_value_cache_main_data[RUNTIME_VALUE_CACHE_MAIN_DATA_SIZE] RUNTIME_SEC(".kv_main");
static unsigned char g_key_cache_main_scale[(RUNTIME_KEY_CACHE_MAIN_SCALE_SIZE > 0u) ? RUNTIME_KEY_CACHE_MAIN_SCALE_SIZE : 1u] RUNTIME_SEC(".kv_main");
static unsigned char g_value_cache_main_scale[(RUNTIME_VALUE_CACHE_MAIN_SCALE_SIZE > 0u) ? RUNTIME_VALUE_CACHE_MAIN_SCALE_SIZE : 1u] RUNTIME_SEC(".kv_main");
static unsigned char g_accel_scratch[RUNTIME_ACCEL_SCRATCH_SIZE] RUNTIME_SEC(".accel_shared");
static unsigned char g_accel_trace[RUNTIME_ACCEL_TRACE_SIZE] RUNTIME_SEC(".accel_trace");

// command window backing storage：
// - 这块故意不复用现有 1MB shared window；
// - 目的是把 queue/control plane 和 data plane 解耦；
// - 当前先作为 host/HW_STUB 下的静态 backing storage，后续 SoC 上可映射到独立 SRAM 窗口。
static unsigned char g_accel_cmdq[RUNTIME_ACCEL_CMDQ_SIZE] RUNTIME_SEC(".accel_cmdq");
static unsigned char g_accel_cmpq[RUNTIME_ACCEL_CMPQ_SIZE] RUNTIME_SEC(".accel_cmpq");
static unsigned char g_accel_dbg2[RUNTIME_ACCEL_DBG2_SIZE] RUNTIME_SEC(".accel_dbg2");

typedef struct {
    uint32_t mmio_base_phys;
    uintptr_t mmio_base_uncached;
    SharedBufferDesc shared_in;
    SharedBufferDesc shared_out;
    SharedBufferDesc shared_param;
    SharedBufferDesc key_cache_main_data;
    SharedBufferDesc key_cache_main_scale;
    SharedBufferDesc value_cache_main_data;
    SharedBufferDesc value_cache_main_scale;
    SharedBufferDesc shared_scratch;
    SharedBufferDesc shared_trace;
    SharedBufferDesc shared_cmdq;
    SharedBufferDesc shared_cmpq;
    SharedBufferDesc shared_dbg2;
    // queue-shadow 状态：
    // - 先按 ring queue 的 head/tail 单调计数器管理；
    // - 当前仍保留 legacy submit 外壳，但 completion 语义统一改成 seq_id。
    uint32_t cmdq_tail;
    uint32_t cmdq_head;
    uint32_t cmpq_tail;
    uint32_t cmpq_head;
    uint32_t next_seq_id;
    uint32_t next_trace_tag;
} RuntimeHwAdapter;

static RuntimeHwAdapter g_hw_adapter = {
    RUNTIME_ACCEL_MMIO_BASE_PHYS,
    RUNTIME_ACCEL_MMIO_BASE_UNC,
    {g_accel_io_in, RUNTIME_ACCEL_IO_IN_UNC, RUNTIME_ACCEL_IO_IN_PHYS, RUNTIME_ACCEL_IO_IN_SIZE},
    {g_accel_io_out, RUNTIME_ACCEL_IO_OUT_UNC, RUNTIME_ACCEL_IO_OUT_PHYS, RUNTIME_ACCEL_IO_OUT_SIZE},
    {g_accel_param, RUNTIME_ACCEL_PARAM_UNC, RUNTIME_ACCEL_PARAM_PHYS, RUNTIME_ACCEL_PARAM_SIZE},
    {g_key_cache_main_data, RUNTIME_KEY_CACHE_MAIN_DATA_UNC, RUNTIME_KEY_CACHE_MAIN_DATA_PHYS, RUNTIME_KEY_CACHE_MAIN_DATA_SIZE},
    {g_key_cache_main_scale, RUNTIME_KEY_CACHE_MAIN_SCALE_UNC, RUNTIME_KEY_CACHE_MAIN_SCALE_PHYS, RUNTIME_KEY_CACHE_MAIN_SCALE_SIZE},
    {g_value_cache_main_data, RUNTIME_VALUE_CACHE_MAIN_DATA_UNC, RUNTIME_VALUE_CACHE_MAIN_DATA_PHYS, RUNTIME_VALUE_CACHE_MAIN_DATA_SIZE},
    {g_value_cache_main_scale, RUNTIME_VALUE_CACHE_MAIN_SCALE_UNC, RUNTIME_VALUE_CACHE_MAIN_SCALE_PHYS, RUNTIME_VALUE_CACHE_MAIN_SCALE_SIZE},
    {g_accel_scratch, RUNTIME_ACCEL_SCRATCH_UNC, RUNTIME_ACCEL_SCRATCH_PHYS, RUNTIME_ACCEL_SCRATCH_SIZE},
    {g_accel_trace, RUNTIME_ACCEL_TRACE_UNC, RUNTIME_ACCEL_TRACE_PHYS, RUNTIME_ACCEL_TRACE_SIZE},
    {g_accel_cmdq, RUNTIME_ACCEL_CMDQ_UNC, RUNTIME_ACCEL_CMDQ_PHYS, RUNTIME_ACCEL_CMDQ_SIZE},
    {g_accel_cmpq, RUNTIME_ACCEL_CMPQ_UNC, RUNTIME_ACCEL_CMPQ_PHYS, RUNTIME_ACCEL_CMPQ_SIZE},
    {g_accel_dbg2, RUNTIME_ACCEL_DBG2_UNC, RUNTIME_ACCEL_DBG2_PHYS, RUNTIME_ACCEL_DBG2_SIZE},
    0u,
    0u,
    0u,
    0u,
    1u,
    1u,
};

static int check_dma_args(const SharedBufferDesc *buf, size_t offset, size_t bytes, const char *tag) {
    if (!buf || !buf->cpu_ptr) {
        fprintf(stderr, "%s: 共享缓冲为空，无法执行 DMA 操作\n", tag);
        return -1;
    }
    if (offset > buf->size || bytes > (buf->size - offset)) {
        fprintf(stderr, "%s: 越界 offset=0x%zx bytes=0x%zx size=0x%zx\n", tag, offset, bytes, buf->size);
        return -1;
    }
    return 0;
}

static uint8_t clamp_qos_class(uint8_t qos_class) {
    // 当前 RTL dma_rdwr 只定义了 2-bit qos class。
    // 软件侧先在 builder 里收口，避免后面 queue mode 联调时出现隐式截断。
    return (uint8_t)(qos_class & 0x3u);
}

static int cmdq_shadow_push(const RuntimeHwCmdEntry *entry, uint32_t *slot_idx_out) {
    size_t depth = runtime_hw_cmdq_capacity();
    size_t slot_idx;
    unsigned char *slot_ptr;

    if (!entry || depth == 0u) {
        return -1;
    }
    if ((g_hw_adapter.cmdq_tail - g_hw_adapter.cmdq_head) >= (uint32_t)depth) {
        fprintf(stderr, "cmdq_shadow_push: CMDQ 已满 tail=%u head=%u depth=%zu\n",
            g_hw_adapter.cmdq_tail,
            g_hw_adapter.cmdq_head,
            depth);
        return -1;
    }

    slot_idx = (size_t)(g_hw_adapter.cmdq_tail % (uint32_t)depth);
    slot_ptr = (unsigned char *)g_hw_adapter.shared_cmdq.cpu_ptr +
               (slot_idx * RUNTIME_HW_CMD_ENTRY_SIZE);
    memcpy(slot_ptr, entry, sizeof(*entry));

    if (slot_idx_out) {
        *slot_idx_out = (uint32_t)slot_idx;
    }

    g_hw_adapter.cmdq_tail = g_hw_adapter.cmdq_tail + 1u;
    return 0;
}

static int cmpq_shadow_push(const RuntimeHwCmpEntry *entry) {
    size_t depth = runtime_hw_cmpq_capacity();
    size_t slot_idx;
    unsigned char *slot_ptr;

    if (!entry || depth == 0u) {
        return -1;
    }
    if ((g_hw_adapter.cmpq_tail - g_hw_adapter.cmpq_head) >= (uint32_t)depth) {
        fprintf(stderr, "cmpq_shadow_push: CMPQ 已满 tail=%u head=%u depth=%zu\n",
            g_hw_adapter.cmpq_tail,
            g_hw_adapter.cmpq_head,
            depth);
        return -1;
    }

    slot_idx = (size_t)(g_hw_adapter.cmpq_tail % (uint32_t)depth);
    slot_ptr = (unsigned char *)g_hw_adapter.shared_cmpq.cpu_ptr +
               (slot_idx * RUNTIME_HW_CMP_ENTRY_SIZE);
    memcpy(slot_ptr, entry, sizeof(*entry));
    g_hw_adapter.cmpq_tail = g_hw_adapter.cmpq_tail + 1u;
    return 0;
}

static int cmpq_shadow_pop(RuntimeHwCmpEntry *entry_out) {
    size_t depth = runtime_hw_cmpq_capacity();
    size_t slot_idx;
    unsigned char *slot_ptr;

    if (!entry_out || depth == 0u) {
        return -1;
    }
    if (g_hw_adapter.cmpq_tail == g_hw_adapter.cmpq_head) {
        return -1;
    }

    slot_idx = (size_t)(g_hw_adapter.cmpq_head % (uint32_t)depth);
    slot_ptr = (unsigned char *)g_hw_adapter.shared_cmpq.cpu_ptr +
               (slot_idx * RUNTIME_HW_CMP_ENTRY_SIZE);
    memcpy(entry_out, slot_ptr, sizeof(*entry_out));
    g_hw_adapter.cmpq_head = g_hw_adapter.cmpq_head + 1u;
    g_hw_adapter.cmdq_head = g_hw_adapter.cmdq_head + 1u;
    return 0;
}

static int queue_shadow_is_empty(void) {
    return (g_hw_adapter.cmdq_tail == g_hw_adapter.cmdq_head) &&
           (g_hw_adapter.cmpq_tail == g_hw_adapter.cmpq_head);
}

static int queue_shadow_is_valid(void) {
    size_t cmdq_depth = runtime_hw_cmdq_capacity();
    size_t cmpq_depth = runtime_hw_cmpq_capacity();
    if (cmdq_depth == 0u || cmpq_depth == 0u) {
        return 0;
    }
    if ((g_hw_adapter.cmdq_tail - g_hw_adapter.cmdq_head) > (uint32_t)cmdq_depth) {
        return 0;
    }
    if ((g_hw_adapter.cmpq_tail - g_hw_adapter.cmpq_head) > (uint32_t)cmpq_depth) {
        return 0;
    }
    return 1;
}

static void trace_queue_shadow_submit(const RuntimeHwLinearJob *job) {
    if (!job) {
        return;
    }
    fprintf(stdout,
        "[HW_ADAPTER] queue_shadow op=%s layer=%d seq_id=%u trace_tag=%u cmdq_slot=%u cmd_count=%u\n",
        job->op_name ? job->op_name : "unknown",
        job->layer_idx,
        job->seq_id,
        job->trace_tag,
        job->cmdq_shadow_offset,
        job->cmdq_shadow_count);
}

static void trace_cmpq_completion(const RuntimeHwCmpEntry *entry) {
    if (!entry) {
        return;
    }
    fprintf(stdout,
        "[HW_ADAPTER] completion seq_id=%u status=%u error=%u cycles=%u info0=%u\n",
        entry->seq_id,
        (unsigned)entry->status,
        (unsigned)entry->error_code,
        entry->cycles,
        entry->info0);
}


void runtime_hw_adapter_init(void) {
    // 第一版不做真实硬件初始化，仅清零共享区，保证每次启动状态可复现。
    memset(g_accel_io_in, 0, sizeof(g_accel_io_in));
    memset(g_accel_io_out, 0, sizeof(g_accel_io_out));
    memset(g_accel_param, 0, sizeof(g_accel_param));
    memset(g_key_cache_main_data, 0, sizeof(g_key_cache_main_data));
    memset(g_value_cache_main_data, 0, sizeof(g_value_cache_main_data));
    memset(g_key_cache_main_scale, 0, sizeof(g_key_cache_main_scale));
    memset(g_value_cache_main_scale, 0, sizeof(g_value_cache_main_scale));
    memset(g_accel_scratch, 0, sizeof(g_accel_scratch));
    memset(g_accel_trace, 0, sizeof(g_accel_trace));
    memset(g_accel_cmdq, 0, sizeof(g_accel_cmdq));
    memset(g_accel_cmpq, 0, sizeof(g_accel_cmpq));
    memset(g_accel_dbg2, 0, sizeof(g_accel_dbg2));
    g_hw_adapter.cmdq_tail = 0u;
    g_hw_adapter.cmdq_head = 0u;
    g_hw_adapter.cmpq_tail = 0u;
    g_hw_adapter.cmpq_head = 0u;
    g_hw_adapter.next_seq_id = 1u;
    g_hw_adapter.next_trace_tag = 1u;
}

void runtime_hw_adapter_dump_layout(FILE *fp) {
    // 布局打印既服务 bring-up，也服务文档化核对：
    // 通过一次输出同时确认 ptr / uncached / phys / size 四套信息是否一致。
    fprintf(fp, "[HW_ADAPTER] MMIO phys=0x%08x uncached=0x%08llx\n",
        g_hw_adapter.mmio_base_phys,
        (unsigned long long)g_hw_adapter.mmio_base_uncached);
    fprintf(fp, "[HW_ADAPTER] IN      ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_in.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_in.cpu_uncached_addr,
        g_hw_adapter.shared_in.phys_addr,
        g_hw_adapter.shared_in.size);
    fprintf(fp, "[HW_ADAPTER] OUT     ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_out.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_out.cpu_uncached_addr,
        g_hw_adapter.shared_out.phys_addr,
        g_hw_adapter.shared_out.size);
    fprintf(fp, "[HW_ADAPTER] PARAM   ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_param.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_param.cpu_uncached_addr,
        g_hw_adapter.shared_param.phys_addr,
        g_hw_adapter.shared_param.size);
    fprintf(fp, "[HW_ADAPTER] KEYDAT  ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.key_cache_main_data.cpu_ptr,
        (unsigned long long)g_hw_adapter.key_cache_main_data.cpu_uncached_addr,
        g_hw_adapter.key_cache_main_data.phys_addr,
        g_hw_adapter.key_cache_main_data.size);
    fprintf(fp, "[HW_ADAPTER] KEYSCL  ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.key_cache_main_scale.cpu_ptr,
        (unsigned long long)g_hw_adapter.key_cache_main_scale.cpu_uncached_addr,
        g_hw_adapter.key_cache_main_scale.phys_addr,
        g_hw_adapter.key_cache_main_scale.size);
    fprintf(fp, "[HW_ADAPTER] VALDAT  ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.value_cache_main_data.cpu_ptr,
        (unsigned long long)g_hw_adapter.value_cache_main_data.cpu_uncached_addr,
        g_hw_adapter.value_cache_main_data.phys_addr,
        g_hw_adapter.value_cache_main_data.size);
    fprintf(fp, "[HW_ADAPTER] VALSCL  ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.value_cache_main_scale.cpu_ptr,
        (unsigned long long)g_hw_adapter.value_cache_main_scale.cpu_uncached_addr,
        g_hw_adapter.value_cache_main_scale.phys_addr,
        g_hw_adapter.value_cache_main_scale.size);
    fprintf(fp, "[HW_ADAPTER] SCRATCH ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_scratch.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_scratch.cpu_uncached_addr,
        g_hw_adapter.shared_scratch.phys_addr,
        g_hw_adapter.shared_scratch.size);
    fprintf(fp, "[HW_ADAPTER] TRACE   ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_trace.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_trace.cpu_uncached_addr,
        g_hw_adapter.shared_trace.phys_addr,
        g_hw_adapter.shared_trace.size);
    fprintf(fp, "[HW_ADAPTER] CMDQ    ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_cmdq.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_cmdq.cpu_uncached_addr,
        g_hw_adapter.shared_cmdq.phys_addr,
        g_hw_adapter.shared_cmdq.size);
    fprintf(fp, "[HW_ADAPTER] CMPQ    ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_cmpq.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_cmpq.cpu_uncached_addr,
        g_hw_adapter.shared_cmpq.phys_addr,
        g_hw_adapter.shared_cmpq.size);
    fprintf(fp, "[HW_ADAPTER] DBG2    ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_dbg2.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_dbg2.cpu_uncached_addr,
        g_hw_adapter.shared_dbg2.phys_addr,
        g_hw_adapter.shared_dbg2.size);
    fprintf(fp, "[HW_ADAPTER] CMDQ entry_size=0x%zx depth=%zu | CMPQ entry_size=0x%zx depth=%zu\n",
        RUNTIME_HW_CMD_ENTRY_SIZE,
        runtime_hw_cmdq_capacity(),
        RUNTIME_HW_CMP_ENTRY_SIZE,
        runtime_hw_cmpq_capacity());
}

void runtime_hw_adapter_trace_op(const char *op_name, int layer_idx, int elem_count) {
    fprintf(stdout, "[HW_ADAPTER] op=%s layer=%d elems=%d in_cpu=0x%08llx out_cpu=0x%08llx in_phys=0x%08x out_phys=0x%08x\n",
        op_name,
        layer_idx,
        elem_count,
        (unsigned long long)g_hw_adapter.shared_in.cpu_uncached_addr,
        (unsigned long long)g_hw_adapter.shared_out.cpu_uncached_addr,
        g_hw_adapter.shared_in.phys_addr,
        g_hw_adapter.shared_out.phys_addr);
}

int runtime_hw_prepare_linear_job(RuntimeHwLinearJob *job, const char *op_name, int layer_idx, size_t elem_count) {
    RuntimeHwCmdEntry cmd_entry;
    uint16_t elem_bytes;
    uint32_t slot_idx;

    if (!job || !op_name) return -1;
    // 这里做的是“语义级线性算子 -> 最小硬件作业描述”的映射，
    // 目标是先冻结接口形状，再逐步替换底层提交细节。
    // 线性算子统一绑定 in/out/param 三块区域。
    // 后续接入真实硬件时，这里可以继续扩展 bank、stride、quant 等字段。
    job->op_name = op_name;
    job->layer_idx = layer_idx;
    job->elem_count = elem_count;
    job->in = &g_hw_adapter.shared_in;
    job->out = &g_hw_adapter.shared_out;
    job->param = &g_hw_adapter.shared_param;
    job->seq_id = g_hw_adapter.next_seq_id++;
    job->trace_tag = g_hw_adapter.next_trace_tag++;
    job->cmdq_shadow_offset = 0u;
    job->cmdq_shadow_count = 0u;

    // queue-shadow 仅固化最小 DMA_READ 组包路径：
    // - 不改默认执行语义；
    // - 先让软件侧具备与 RuntimeHwCmdEntry 对齐的可观测 enqueue 行为。
    elem_bytes = (uint16_t)((elem_count > 0xffffu) ? 0xffffu : (uint16_t)elem_count);
    if (runtime_hw_init_cmd_entry(
            &cmd_entry,
            RUNTIME_HW_CMD_DMA_READ,
            job->seq_id,
            job->trace_tag,
            job->in->phys_addr,
            elem_bytes,
            0u,
            0u,
            0u,
            0u) != 0) {
        return -1;
    }

    if (cmdq_shadow_push(&cmd_entry, &slot_idx) != 0) {
        return -1;
    }

    job->cmdq_shadow_offset = slot_idx;
    job->cmdq_shadow_count = 1u;
    return 0;
}

int runtime_hw_prepare_post_job(RuntimeHwPostJob *job, const char *op_name, int layer_idx, size_t elem_count) {
    if (!job || !op_name) return -1;
    // 后处理算子统一绑定 in/out/scratch，便于 softmax/gate/residual 一类算子复用。
    job->op_name = op_name;
    job->layer_idx = layer_idx;
    job->elem_count = elem_count;
    job->in = &g_hw_adapter.shared_in;
    job->out = &g_hw_adapter.shared_out;
    job->scratch = &g_hw_adapter.shared_scratch;
    return 0;
}

int runtime_hw_dma_load(const SharedBufferDesc *dst, size_t dst_offset, const void *src, size_t bytes) {
    if (!src) return -1;
    if (check_dma_args(dst, dst_offset, bytes, "runtime_hw_dma_load") != 0) return -1;
    // host/HW_STUB 下用 memcpy 模拟 DMA，保持边界检查和接口语义一致。
    memcpy((unsigned char *)dst->cpu_ptr + dst_offset, src, bytes);
    return 0;
}

int runtime_hw_dma_store(void *dst, const SharedBufferDesc *src, size_t src_offset, size_t bytes) {
    if (!dst) return -1;
    if (check_dma_args(src, src_offset, bytes, "runtime_hw_dma_store") != 0) return -1;
    memcpy(dst, (const unsigned char *)src->cpu_ptr + src_offset, bytes);
    return 0;
}

int runtime_hw_submit_job(const char *job_kind, const void *job) {
    const RuntimeHwLinearJob *linear_job = (const RuntimeHwLinearJob *)job;
    RuntimeHwCmpEntry cmp_entry;

    // 这里保留“提交”语义边界。真实硬件版本会在此处：
    // 1) 写 MMIO 描述符；
    //    或者把 descriptor enqueue 到独立的 CMDQ 窗口；
    // 2) 触发启动；
    // 3) 记录 seq_id / trace_tag。
    if (!job_kind || !job) return -1;
    fprintf(stdout, "[HW_ADAPTER] submit job_kind=%s\n", job_kind);
    if (strcmp(job_kind, "linear") == 0) {
        trace_queue_shadow_submit(linear_job);
        if (runtime_hw_init_cmp_entry(&cmp_entry, linear_job->seq_id, RUNTIME_HW_CMP_DONE, 0u, 0u, linear_job->trace_tag) != 0) {
            return -1;
        }
        if (cmpq_shadow_push(&cmp_entry) != 0) {
            return -1;
        }
    }
    return 0;
}

int runtime_hw_wait_done(uint32_t timeout_cycle) {
    RuntimeHwCmpEntry cmp_entry;
    // 第一版先消费 queue-shadow completion；
    // timeout 参数保留，后续接真实 doorbell / wait path 时不改上层约定。
    (void)timeout_cycle;
    if (cmpq_shadow_pop(&cmp_entry) != 0) {
        fprintf(stderr, "runtime_hw_wait_done: 当前没有可消费的 completion\n");
        return -1;
    }
    trace_cmpq_completion(&cmp_entry);
    return (cmp_entry.status == (uint16_t)RUNTIME_HW_CMP_DONE) ? 0 : -1;
}

void runtime_hw_soft_reset(void) {
    // 软复位时清空共享区，模拟硬件 reset 后的干净状态。
    runtime_hw_adapter_init();
}

int runtime_hw_init_cmd_entry(
    RuntimeHwCmdEntry *entry,
    RuntimeHwCmdOpcode opcode,
    uint32_t seq_id,
    uint32_t trace_tag,
    uint32_t addr,
    uint16_t len_bytes,
    uint8_t qos_class,
    uint8_t stride_en,
    uint16_t line_size,
    uint16_t line_stride
) {
    if (!entry) return -1;

    // 这里冻结的是“最小 command 合同”，字段直接贴合当前 dma_rdwr：
    // - opcode      <-> cmd_rw_i
    // - addr        <-> cmd_addr_i
    // - len_bytes   <-> cmd_len_bytes_i
    // - qos_class   <-> cmd_qos_class_i
    // - stride_en   <-> cmd_stride_en_i
    // - line_size   <-> cmd_line_size_i
    // - line_stride <-> cmd_line_stride_i
    // 同时把 completion 身份语义统一到 seq_id，trace_tag 仅作软件观测。
    memset(entry, 0, sizeof(*entry));
    entry->opcode = (uint8_t)opcode;
    entry->flags = 0;
    entry->qos_class = clamp_qos_class(qos_class);
    entry->addr = addr;
    entry->len_bytes = len_bytes;
    entry->stride_en = (uint8_t)(stride_en ? 1u : 0u);
    entry->line_size = line_size;
    entry->line_stride = line_stride;
    entry->seq_id = seq_id;
    entry->user0 = trace_tag;
    return 0;
}

int runtime_hw_init_cmp_entry(
    RuntimeHwCmpEntry *entry,
    uint32_t seq_id,
    RuntimeHwCmpStatus status,
    uint16_t error_code,
    uint32_t cycles,
    uint32_t info0
) {
    if (!entry) return -1;
    memset(entry, 0, sizeof(*entry));
    entry->seq_id = seq_id;
    entry->status = (uint16_t)status;
    entry->error_code = error_code;
    entry->cycles = cycles;
    entry->info0 = info0;
    return 0;
}

size_t runtime_hw_cmdq_capacity(void) {
    return (size_t)(RUNTIME_ACCEL_CMDQ_SIZE / RUNTIME_HW_CMD_ENTRY_SIZE);
}

size_t runtime_hw_cmpq_capacity(void) {
    return (size_t)(RUNTIME_ACCEL_CMPQ_SIZE / RUNTIME_HW_CMP_ENTRY_SIZE);
}

int runtime_hw_queue_layout_is_valid(void) {
    // 这个检查主要给 bring-up / assert / 单测用：
    // 只验证 queue window 和 entry 大小是否自洽，以及 ring 生命周期是否仍在合法范围。
    if ((RUNTIME_ACCEL_CMDQ_SIZE % RUNTIME_HW_CMD_ENTRY_SIZE) != 0) return 0;
    if ((RUNTIME_ACCEL_CMPQ_SIZE % RUNTIME_HW_CMP_ENTRY_SIZE) != 0) return 0;
    if (runtime_hw_cmdq_capacity() == 0) return 0;
    if (runtime_hw_cmpq_capacity() == 0) return 0;
    if (!queue_shadow_is_valid()) return 0;
    return 1;
}

const SharedBufferDesc *runtime_hw_adapter_shared_in(void) {
    return &g_hw_adapter.shared_in;
}

const SharedBufferDesc *runtime_hw_adapter_shared_out(void) {
    return &g_hw_adapter.shared_out;
}

const SharedBufferDesc *runtime_hw_adapter_shared_param(void) {
    return &g_hw_adapter.shared_param;
}

const SharedBufferDesc *runtime_hw_adapter_shared_kv(void) {
    return &g_hw_adapter.key_cache_main_data;
}

const SharedBufferDesc *runtime_hw_adapter_key_cache_main_data(void) {
    return &g_hw_adapter.key_cache_main_data;
}

const SharedBufferDesc *runtime_hw_adapter_key_cache_main_scale(void) {
    return &g_hw_adapter.key_cache_main_scale;
}

const SharedBufferDesc *runtime_hw_adapter_value_cache_main_data(void) {
    return &g_hw_adapter.value_cache_main_data;
}

const SharedBufferDesc *runtime_hw_adapter_value_cache_main_scale(void) {
    return &g_hw_adapter.value_cache_main_scale;
}

uint32_t runtime_hw_adapter_cmdq_head(void) {
    return g_hw_adapter.cmdq_head;
}

uint32_t runtime_hw_adapter_cmdq_tail(void) {
    return g_hw_adapter.cmdq_tail;
}

uint32_t runtime_hw_adapter_cmpq_head(void) {
    return g_hw_adapter.cmpq_head;
}

uint32_t runtime_hw_adapter_cmpq_tail(void) {
    return g_hw_adapter.cmpq_tail;
}

int runtime_hw_adapter_queue_shadow_is_empty(void) {
    return queue_shadow_is_empty();
}

const SharedBufferDesc *runtime_hw_adapter_shared_scratch(void) {
    return &g_hw_adapter.shared_scratch;
}

const SharedBufferDesc *runtime_hw_adapter_shared_trace(void) {
    return &g_hw_adapter.shared_trace;
}

const SharedBufferDesc *runtime_hw_adapter_shared_cmdq(void) {
    return &g_hw_adapter.shared_cmdq;
}

const SharedBufferDesc *runtime_hw_adapter_shared_cmpq(void) {
    return &g_hw_adapter.shared_cmpq;
}

const SharedBufferDesc *runtime_hw_adapter_shared_dbg2(void) {
    return &g_hw_adapter.shared_dbg2;
}
