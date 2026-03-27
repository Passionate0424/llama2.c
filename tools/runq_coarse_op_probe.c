#define TESTING
#include "../runq.c"

/*
 * runq_coarse_op_probe.c
 *
 * 作用：
 * 1. 复用 runq.c 里的 checkpoint 解析和 Config/Transformer 定义
 * 2. 不做真正的 token 推理，不跑 forward()/generate()
 * 3. 只根据模型配置，把 runq 的线性层拆成“粗粒度 op”清单
 * 4. 输出每类 op 的形状、MAC 数、输入/输出字节数，帮助设计硬件软件契约
 *
 * 它更像一个“静态 schedule / MAC 覆盖率分析器”，而不是推理程序。
 *
 * 使用方法：
 *   runq_coarse_op_probe <checkpoint.bin> [chunk_n]
 *
 * 示例：
 *   runq_coarse_op_probe artifacts/stories260K/stories260K_q80.bin 4
 *
 * 这里的 chunk_n 用来模拟：
 * - decode: chunk_n = 1
 * - prefill: chunk_n = 4 / 8
 *
 * 输出内容：
 * - CONFIG: 模型超参数
 * - OP: 每个粗粒度线性算子的 m/k/n、MAC 和张量字节数
 * - SUMMARY: attention / ffn_in / ffn_out / classifier 的 MAC 占比
 */

static long long macs_3d(int m, int k, int n) {
    return (long long)m * (long long)k * (long long)n;
}

static void print_op_line(
    int layer,
    const char* name,
    int m,
    int k,
    int n,
    const char* family
) {
    /* 这里输出的是“如果把一个线性层作为一个粗粒度 op 提交给硬件”时，
     * 软件/硬件最关心的静态信息：
     * - m/k/n
     * - MAC 数量
     * - 权重/输入/ACC32 输出的字节数
     */
    long long macs = macs_3d(m, k, n);
    long long w_bytes = (long long)m * (long long)k;
    long long x_bytes = (long long)k * (long long)n;
    long long y_acc_bytes = (long long)m * (long long)n * 4LL;
    printf(
        "OP layer=%d name=%s family=%s m=%d k=%d n=%d macs=%lld weight_bytes=%lld input_bytes=%lld acc32_bytes=%lld\n",
        layer,
        name,
        family,
        m,
        k,
        n,
        macs,
        w_bytes,
        x_bytes,
        y_acc_bytes
    );
}

static void emit_schedule(const Config* p, int chunk_n) {
    /* runq 的线性层族：
     * - attention: wq/wk/wv/wo
     * - ffn_in   : w1/w3
     * - ffn_out  : w2
     * - classifier: wcls
     *
     * 这里按粗粒度 op 视角，把每层要提交给硬件的线性算子都枚举出来。
     */
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    long long attn_macs = 0;
    long long ffn_in_macs = 0;
    long long ffn_out_macs = 0;
    long long cls_macs = 0;

    printf(
        "CONFIG dim=%d hidden_dim=%d layers=%d heads=%d kv_heads=%d vocab=%d seq_len=%d gs=%d chunk_n=%d\n",
        p->dim,
        p->hidden_dim,
        p->n_layers,
        p->n_heads,
        p->n_kv_heads,
        p->vocab_size,
        p->seq_len,
        GS,
        chunk_n
    );

    for (int l = 0; l < p->n_layers; l++) {
        /* 注意：
         * 这里的 n=chunk_n 不是 runq.c 当前 decode 的真实执行宽度，
         * 而是用来模拟“如果软件把 token 组织成 N 列一起提交给硬件”
         * 时，硬件看到的 GEMM 形状。
         */
        print_op_line(l, "wq", p->dim, p->dim, chunk_n, "attn");
        print_op_line(l, "wk", kv_dim, p->dim, chunk_n, "attn");
        print_op_line(l, "wv", kv_dim, p->dim, chunk_n, "attn");
        print_op_line(l, "wo", p->dim, p->dim, chunk_n, "attn");
        print_op_line(l, "w1", p->hidden_dim, p->dim, chunk_n, "ffn_in");
        print_op_line(l, "w3", p->hidden_dim, p->dim, chunk_n, "ffn_in");
        print_op_line(l, "w2", p->dim, p->hidden_dim, chunk_n, "ffn_out");

        attn_macs += macs_3d(p->dim, p->dim, chunk_n);
        attn_macs += macs_3d(kv_dim, p->dim, chunk_n);
        attn_macs += macs_3d(kv_dim, p->dim, chunk_n);
        attn_macs += macs_3d(p->dim, p->dim, chunk_n);
        ffn_in_macs += macs_3d(p->hidden_dim, p->dim, chunk_n);
        ffn_in_macs += macs_3d(p->hidden_dim, p->dim, chunk_n);
        ffn_out_macs += macs_3d(p->dim, p->hidden_dim, chunk_n);
    }

    print_op_line(-1, "wcls", p->vocab_size, p->dim, chunk_n, "classifier");
    cls_macs = macs_3d(p->vocab_size, p->dim, chunk_n);

    long long linear_total = attn_macs + ffn_in_macs + ffn_out_macs + cls_macs;
    long long no_cls_total = attn_macs + ffn_in_macs + ffn_out_macs;
    long long attn_ffn13 = attn_macs + ffn_in_macs;

    printf(
        "SUMMARY attn_macs=%lld ffn_in_macs=%lld ffn_out_macs=%lld cls_macs=%lld total_linear_macs=%lld total_wo_cls=%lld attn_ffn13_ratio=%.6f w2_ratio=%.6f\n",
        attn_macs,
        ffn_in_macs,
        ffn_out_macs,
        cls_macs,
        linear_total,
        no_cls_total,
        no_cls_total ? (double)attn_ffn13 / (double)no_cls_total : 0.0,
        no_cls_total ? (double)ffn_out_macs / (double)no_cls_total : 0.0
    );
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: runq_coarse_op_probe <checkpoint.bin> [chunk_n]\n");
        return 1;
    }

    int chunk_n = 4;
    if (argc >= 3) {
        chunk_n = atoi(argv[2]);
        if (chunk_n <= 0) {
            chunk_n = 4;
        }
    }

    /* 这里仅仅借用 runq.c 的模型加载逻辑，把模型头和权重映射进来。
     * 不会进行 token 推理，也不会执行 forward()/sample()/generate()。
     */
    Transformer transformer = {0};
    build_transformer(&transformer, argv[1]);

    /* 输出粗粒度 op schedule 和 MAC 覆盖率统计。 */
    emit_schedule(&transformer.config, chunk_n);

    /* 释放 runq.c 里分配的资源。 */
    free_transformer(&transformer);
    return 0;
}
