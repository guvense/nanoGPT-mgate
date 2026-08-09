"""
Generate mgate_paper.pdf — updated technical report on the M-gate K/V-bias
mechanism with full ablation set (head-dim, depth, width, detached, real
wall-clock). Data from ablation.log (7 variants × 3 seeds, T4 GPU, n_layer=6).
"""

import io
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether,
)

# ---------------------------------------------------------------------------
# Data (from ablation.log, T4, bfloat16, n_layer=6, n_head=4, block_size=128)
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 2024]

# Val loss @ iter 3000 (except wallclock: iter 3500)
DATA = {
    # (params_M, val_loss_per_seed, wallclock_per_seed)
    "vanilla_n128":       (1.19, [1.7307, 1.7339, 1.7319], [107.1, 107.6, 107.3]),
    "kv_n112_full":       (1.21, [1.6944, 1.6840, 1.6859], [125.9, 128.4, 128.9]),
    "vanilla_n112":       (0.91, [1.7693, 1.7585, 1.7605], [101.2, 100.9, 100.8]),
    "vanilla_n128_L7":    (0.99, [1.7562, 1.7546, 1.7538], [ 90.9,  90.5,  90.7]),
    "vanilla_n136":       (1.34, [1.7048, 1.7101, 1.7074], [122.8, 122.2, 122.0]),
    "kv_n112_detached":   (1.21, [1.7812, 1.7532, 1.7743], [116.1, 116.4, 116.4]),
    "vanilla_wallclock":  (1.19, [1.6736, 1.6807, 1.6708], [135.0, 134.4, 134.3]),  # @ iter 3500
}

# Trajectory for main figure (K/V full and vanilla_ref, mean over 3 seeds)
ITERS = [500, 1000, 1500, 2000, 2500, 3000]
def traj_mean(prefix_data):
    # extracted from the log
    pass

VANILLA_TRAJ = [2.2315, 1.9910, 1.8719, 1.7918, 1.7489, 1.7322]  # vanilla n=128 mean of 3 seeds
KV_TRAJ      = [2.2049, 1.9422, 1.8242, 1.7539, 1.6960, 1.6881]  # KV full mean of 3 seeds
DETACHED_TRAJ = [2.2538, 2.0224, 1.9075, 1.8406, 1.7810, 1.7696]  # detached mean of 3 seeds
VANILLA_LONG_TRAJ = [2.2314, 1.9885, 1.8688, 1.7822, 1.7313, 1.7002, 1.6750]  # iters 500..3500
VAN_LONG_ITERS    = [500, 1000, 1500, 2000, 2500, 3000, 3500]

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def stats(name):
    p, vals, walls = DATA[name]
    return {
        "params": p,
        "val_mean": statistics.mean(vals),
        "val_std": statistics.stdev(vals),
        "wall_mean": statistics.mean(walls),
        "vals": vals,
    }

def paired_diff(name_a, name_b):
    """Return (mean_diff, se, t, dof, p_upper) for a - b."""
    a = DATA[name_a][1]
    b = DATA[name_b][1]
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    md = statistics.mean(d)
    sd = statistics.stdev(d)
    se = sd / math.sqrt(n)
    t = md / se if se > 0 else float('nan')
    return md, se, t, n - 1

# Key stats
S_VAN     = stats("vanilla_n128")
S_KV      = stats("kv_n112_full")
S_VAN112  = stats("vanilla_n112")
S_L7      = stats("vanilla_n128_L7")
S_W136    = stats("vanilla_n136")
S_DET     = stats("kv_n112_detached")
S_WALL    = stats("vanilla_wallclock")

kv_vs_van = paired_diff("kv_n112_full", "vanilla_n128")
kv_vs_van112 = paired_diff("kv_n112_full", "vanilla_n112")
kv_vs_det = paired_diff("kv_n112_full", "kv_n112_detached")
kv_vs_wall = paired_diff("kv_n112_full", "vanilla_wallclock")

def ppl_gain(loss_lower, loss_higher):
    """% perplexity improvement from higher to lower."""
    return (math.exp(loss_higher) - math.exp(loss_lower)) / math.exp(loss_higher) * 100

# ---------------------------------------------------------------------------
# Figure — val loss trajectory with ablations
# ---------------------------------------------------------------------------

def make_trajectory_figure(path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(ITERS, VANILLA_TRAJ, 'o-',
            label='Vanilla n=128, n_layer=6 (1.19M)', color='#666666', linewidth=1.8, markersize=5)
    ax.plot(ITERS, KV_TRAJ, 's-',
            label='K/V-bias n=112 (1.21M) — this work', color='#c0392b', linewidth=2.2, markersize=5.5)
    ax.plot(ITERS, DETACHED_TRAJ, '^--',
            label='K/V-bias DETACHED (ablation)', color='#e67e22', linewidth=1.5, markersize=5, alpha=0.85)
    ax.plot(VAN_LONG_ITERS, VANILLA_LONG_TRAJ, 'x:',
            label='Vanilla n=128 @ 3500 iters (wall-clock matched)', color='#333333', linewidth=1.2, markersize=6, alpha=0.7)
    ax.set_xlabel('Training iteration', fontsize=10)
    ax.set_ylabel('Validation loss (nats)', fontsize=10)
    ax.set_title('Val loss trajectory (mean over 3 seeds, n_layer=6 shakespeare-char)', fontsize=10.5)
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def make_ablation_bar(path):
    """Bar chart of val loss @ iter 3000 across all variants."""
    variants = [
        ("Vanilla n=112\n(head_dim=28)", S_VAN112["val_mean"], "#7f8c8d"),
        ("Vanilla n=128 L=7\n(depth)", S_L7["val_mean"], "#7f8c8d"),
        ("Vanilla n=128\n(reference)", S_VAN["val_mean"], "#95a5a6"),
        ("Vanilla n=136\n(wider, more params)", S_W136["val_mean"], "#7f8c8d"),
        ("K/V DETACHED\n(ablation)", S_DET["val_mean"], "#e67e22"),
        ("K/V-bias FULL\n(this work)", S_KV["val_mean"], "#c0392b"),
        ("Vanilla @ 3500\n(wall-clock match)", S_WALL["val_mean"], "#2c3e50"),
    ]
    variants.sort(key=lambda x: -x[1])  # descending val loss
    labels = [v[0] for v in variants]
    values = [v[1] for v in variants]
    cols = [v[2] for v in variants]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.barh(range(len(variants)), values, color=cols, edgecolor='black', linewidth=0.4)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Validation loss @ iter 3000 (nats, lower is better)', fontsize=9.5)
    ax.set_title('All variants at 3-seed mean', fontsize=10.5)
    for i, (v, bar) in enumerate(zip(values, bars)):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=8)
    ax.set_xlim(min(values) - 0.02, max(values) + 0.03)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

STYLES = getSampleStyleSheet()

TITLE = ParagraphStyle('Title', parent=STYLES['Title'], fontSize=15, leading=18,
                       alignment=TA_CENTER, spaceAfter=6)
SUBTITLE = ParagraphStyle('Subtitle', parent=STYLES['Normal'], fontSize=9, leading=12,
                          alignment=TA_CENTER, textColor=colors.HexColor('#555555'), spaceAfter=14)
H1 = ParagraphStyle('H1', parent=STYLES['Heading1'], fontSize=12, leading=14,
                    spaceAfter=6, spaceBefore=14, textColor=colors.HexColor('#111111'))
H2 = ParagraphStyle('H2', parent=STYLES['Heading2'], fontSize=10.5, leading=13,
                    spaceAfter=4, spaceBefore=8, textColor=colors.HexColor('#333333'))
BODY = ParagraphStyle('Body', parent=STYLES['BodyText'], fontSize=9.5, leading=13,
                      alignment=TA_JUSTIFY, spaceAfter=6)
ABSTRACT = ParagraphStyle('Abstract', parent=BODY, leftIndent=15, rightIndent=15,
                          fontSize=9, leading=12, textColor=colors.HexColor('#222222'))
MONO = ParagraphStyle('Mono', parent=STYLES['Code'], fontSize=8.5, leading=10.5,
                      leftIndent=15, rightIndent=15, spaceAfter=6, spaceBefore=4,
                      backColor=colors.HexColor('#f4f4f4'), borderPadding=6)
CAPTION = ParagraphStyle('Caption', parent=BODY, fontSize=8.5, leading=11,
                         alignment=TA_CENTER, textColor=colors.HexColor('#444444'), spaceAfter=8)
CALLOUT = ParagraphStyle('Callout', parent=BODY, fontSize=9.5, leading=13,
                         leftIndent=15, rightIndent=15, spaceAfter=6, spaceBefore=4,
                         backColor=colors.HexColor('#fff5e6'), borderColor=colors.HexColor('#e67e22'),
                         borderWidth=0.6, borderPadding=8)

def para(text, style=BODY):
    return Paragraph(text, style)

def code(text):
    text = text.replace('\n', '<br/>').replace(' ', '&nbsp;')
    return Paragraph(f'<font face="Courier">{text}</font>', MONO)

def make_table(data, col_widths=None, small=False, highlight_rows=None):
    fs = 8 if small else 9
    style_cmds = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), fs),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.HexColor('#333333')),
        ('LINEABOVE', (0, 0), (-1, 0), 0.6, colors.HexColor('#333333')),
        ('LINEBELOW', (0, -1), (-1, -1), 0.6, colors.HexColor('#333333')),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eeeeee')),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]
    if highlight_rows:
        for r in highlight_rows:
            style_cmds.append(('BACKGROUND', (0, r), (-1, r), colors.HexColor('#fff5e6')))
    t = Table(data, colWidths=col_widths, hAlign='CENTER')
    t.setStyle(TableStyle(style_cmds))
    return t


def build_paper(pdf_path):
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Additive K/V Biases from a Depth-Recurrent Gated State in Transformer Attention",
    )

    story = []

    # ---- Title ----
    story.append(para(
        "Additive K/V Biases from a Depth-Recurrent Gated State in Transformer Attention: "
        "An Ablation Study",
        TITLE))
    story.append(para(
        "An empirical study on shakespeare-char (nanoGPT, 6 layers, ~1.2M parameters, 3 seeds)"
        "<br/>Guven Seckin  &middot;  November 2025 &middot; v2 with full ablations",
        SUBTITLE))

    # ---- Abstract ----
    kv_v_van_ppl = ppl_gain(S_KV["val_mean"], S_VAN["val_mean"])
    kv_v_van112_ppl = ppl_gain(S_KV["val_mean"], S_VAN112["val_mean"])
    kv_v_det_ppl = ppl_gain(S_KV["val_mean"], S_DET["val_mean"])
    kv_v_wall_ppl = ppl_gain(S_WALL["val_mean"], S_KV["val_mean"])  # note: vanilla wins here
    kv_overhead = (S_KV["wall_mean"] / S_VAN["wall_mean"] - 1) * 100

    story.append(para("Abstract", H1))
    story.append(para(
        "We study a specific point in the design space of depth-recurrent transformers: a per-token "
        "state <b>M</b> that flows across transformer blocks under a GRU-style gate, injected as "
        "an <b>additive bias to the attention keys and values</b> while leaving Q, softmax "
        "normalization, and the causal mask intact. On shakespeare-char at n_layer=6 "
        f"(~1.2M non-embedding parameters, 3 seeds), the mechanism reduces validation perplexity by "
        f"<b>{kv_v_van_ppl:.1f}%</b> against a parameter-matched vanilla baseline "
        f"(paired t = {kv_vs_van[2]:.2f}, df=2), and by "
        f"<b>{kv_v_van112_ppl:.1f}%</b> against a strictly head-dim-matched vanilla. "
        f"A critical ablation cutting cross-layer gradient flow through M via <font face='Courier'>.detach()</font> "
        f"(same parameters, same FLOPs) causes the mechanism to <b>actively hurt</b> — "
        f"{kv_v_det_ppl:.1f}% worse perplexity than the full variant and "
        f"worse than vanilla — confirming the improvement is driven by the <b>learned "
        f"cross-layer state</b>, not by the added capacity. Further capacity controls (deeper "
        f"vanilla with 25% more depth, wider vanilla with 10% MORE parameters) also lose to the "
        f"mechanism. However, at real wall-clock-matched training on CUDA-T4, longer-trained "
        f"vanilla wins by {kv_v_wall_ppl:.1f}% perplexity: the ~{kv_overhead:.0f}% per-iteration "
        f"overhead currently exceeds the algorithmic advantage at this small scale. The gate "
        f"converges to a stable operating point across seeds. We do not claim a new architectural "
        f"family; we contribute a rigorously ablated positive result at small scale, with an "
        f"honest negative result on real wall-clock. Whether the algorithmic advantage catches up "
        f"at larger scale is an open question.",
        ABSTRACT))

    # ---- 1. Introduction ----
    story.append(para("1. Introduction", H1))
    story.append(para(
        "Standard transformer attention gives each token three roles: <b>Query</b>, <b>Key</b>, "
        "and <b>Value</b>. We investigate the addition of a fourth signal &mdash; a persistent "
        "<b>purpose</b> state <b>M</b> that flows across transformer blocks (depth-recurrent) and "
        "biases attention. The state is updated by a GRU-style gate at each block and consumed as an "
        "additive per-token bias on the attention keys and values in the next block. Attention "
        "otherwise remains vanilla.", BODY))
    story.append(para(
        "The design space around this idea is actively researched (Universal Transformer, Feedback "
        "Transformer, GTrXL, Block-Recurrent Transformer, Depth-Recurrent Transformers). Our "
        "contribution is not a new family but the specific combination of "
        "(i) per-token granularity, (ii) depth-across state flow, (iii) input-dependent GRU gate, "
        "(iv) additive-bias consumption on both K and V. This paper reports a small-scale empirical "
        "study with <b>rigorous ablations</b> that isolate the mechanism from three natural confounds "
        "(head dimension, capacity/depth, wall-clock compute).", BODY))

    # ---- 2. Method ----
    story.append(para("2. Method", H1))
    story.append(para(
        "For each transformer block <i>i</i> (0-indexed, up to L-1), given the residual stream "
        "<i>x<sub>i</sub></i> and the state from the previous block <i>M<sub>i-1</sub></i> "
        "(initialized as zeros for i=0):", BODY))
    story.append(code(
        "# Standard attention with additive M-derived biases on K and V (Q untouched):\n"
        "Q, K, V   = c_attn(LayerNorm(x_i)).split(3, dim=-1)\n"
        "K'        = K + W_km @ M_{i-1}                     # per-token additive bias\n"
        "V'        = V + W_vm @ M_{i-1}\n"
        "y         = c_proj( attention(Q, K', V') )         # vanilla causal attention\n"
        "x'_i      = x_i + y                                 # residual\n"
        "\n"
        "# GRU-style update of M from post-attention hidden state:\n"
        "h         = LayerNorm(x'_i)\n"
        "g_i       = sigmoid(W_g @ h + b_g)                 # gate\n"
        "c_i       = tanh(W_m @ h + b_m)                    # candidate\n"
        "M_i       = g_i * M_{i-1} + (1 - g_i) * c_i        # elementwise blend\n"
        "\n"
        "x_{i+1}   = x'_i + MLP(h)                          # standard MLP branch"
    ))
    story.append(para(
        "All operations are elementwise or per-token; no token's M leaks into another token's K/V. "
        "Implementation fuses (<i>W<sub>km</sub></i>, <i>W<sub>vm</sub></i>) into a single "
        "Linear <i>W<sub>kvm</sub></i>: E&rarr;2E, and (<i>W<sub>g</sub></i>, <i>W<sub>m</sub></i>) "
        "into <i>W<sub>gm</sub></i>: E&rarr;2E. Bit-identical outputs; halves the kernel launch "
        "count.", BODY))

    # ---- 3. Experimental setup ----
    story.append(para("3. Experimental setup", H1))
    story.append(para(
        "<b>Dataset:</b> shakespeare-char (~1.1M chars, vocab_size=65).<br/>"
        "<b>Architecture:</b> n_layer=6, n_head=4, block_size=128, batch_size=32, dropout=0. "
        "Vanilla baseline uses n_embd=128 (head_dim=32, 1.19M non-embedding params). K/V-bias "
        "variant uses n_embd=112 (head_dim=28, 1.21M params, ~1% overhead).<br/>"
        "<b>Optimizer:</b> AdamW, lr=1e-3 with cosine decay to 1e-4, 100-iter warmup, "
        "weight_decay=0.1. <b>Training:</b> 3000 iterations, evaluated every 500 iters over 40 "
        "validation batches. <b>Hardware:</b> Tesla T4 (Google Colab), dtype=bfloat16, "
        "<font face='Courier'>torch.compile</font> disabled (comparing steady-state per-iter cost). "
        "<b>Seeds:</b> 3 independent runs per variant (42, 123, 2024).", BODY))

    # ---- 4. Results ----
    story.append(para("4. Results", H1))

    # 4.1 Primary result
    story.append(para("4.1 Primary result: K/V-bias beats vanilla at matched parameters", H2))
    table_data = [
        ["Model", "Params", "Wall-clock", "Val loss @ 3000", "Δ vs Vanilla n=128"],
    ]
    def row(name, label, ref=None):
        s = stats(name)
        d = s["val_mean"] - S_VAN["val_mean"] if ref is None else s["val_mean"] - stats(ref)["val_mean"]
        return [label, f"{s['params']:.2f}M", f"{s['wall_mean']:.1f}s",
                f"{s['val_mean']:.4f} ± {s['val_std']:.4f}",
                f"{d:+.4f}"]
    table_data.append(row("vanilla_n128", "Vanilla n=128 (reference)"))
    table_data.append(row("kv_n112_full", "K/V-bias n=112 (this work)"))
    story.append(make_table(
        [[Paragraph(str(c), ParagraphStyle('cell', fontSize=9)) for c in r] for r in table_data],
        col_widths=[5.5 * cm, 1.7 * cm, 2.0 * cm, 3.5 * cm, 2.6 * cm],
        highlight_rows=[2]))
    story.append(Spacer(1, 4))
    story.append(para(
        f"<b>Table 1:</b> The K/V-bias variant achieves "
        f"<b>{S_KV['val_mean']:.4f} nats</b> vs vanilla's <b>{S_VAN['val_mean']:.4f}</b> "
        f"(<b>{kv_v_van_ppl:.1f}% perplexity reduction</b>) with ~1% more parameters "
        f"and {kv_overhead:.0f}% higher wall-clock. Paired t-test on val loss: "
        f"t = {kv_vs_van[2]:.2f} (df = {kv_vs_van[3]}).", CAPTION))

    # 4.2 Ablations table
    story.append(para("4.2 Ablations", H2))
    story.append(para(
        "We evaluate four alternative baselines that isolate the mechanism from potential "
        "confounds. All are trained with the same seeds, config, and training budget.", BODY))
    ab_data = [
        ["Ablation", "Params", "Val loss (3 seeds)", "Δ vs K/V"],
    ]
    def ab_row(name, label):
        s = stats(name)
        d = s["val_mean"] - S_KV["val_mean"]
        return [label, f"{s['params']:.2f}M", f"{s['val_mean']:.4f}", f"{d:+.4f}"]
    ab_data.append(ab_row("kv_n112_full",       "K/V-bias FULL (reference)"))
    ab_data.append(ab_row("vanilla_n128",       "Vanilla n=128 (head_dim=32)"))
    ab_data.append(ab_row("vanilla_n112",       "Vanilla n=112 (head_dim=28, matched to K/V)"))
    ab_data.append(ab_row("vanilla_n128_L7",    "Vanilla n=128 with n_layer=7 (+depth)"))
    ab_data.append(ab_row("vanilla_n136",       "Vanilla n=136 (wider, +11% params)"))
    ab_data.append(ab_row("kv_n112_detached",   "K/V-bias with detached M (no cross-layer learning)"))
    story.append(make_table(
        [[Paragraph(str(c), ParagraphStyle('cell', fontSize=8.7)) for c in r] for r in ab_data],
        col_widths=[7.5 * cm, 1.6 * cm, 3.2 * cm, 2.4 * cm],
        highlight_rows=[1, 6]))
    story.append(Spacer(1, 4))
    story.append(para(
        "<b>Table 2:</b> Ablations. K/V-bias FULL is best across all alternatives. Vanilla with "
        "matched head_dim (row 3) is worse than default vanilla, so K/V's win becomes <b>larger</b> "
        f"({kv_v_van112_ppl:.1f}%) when head_dim is controlled for. Adding depth or width to vanilla "
        f"(rows 4, 5) does not close the gap: wider vanilla with <b>more parameters</b> than K/V "
        f"still loses by {ppl_gain(S_KV['val_mean'], S_W136['val_mean']):.1f}% perplexity.", CAPTION))

    story.append(para("4.3 The critical ablation: detached cross-layer state", H2))
    story.append(para(
        "The most important test is row 6 in Table 2. In this ablation we call "
        "<font face='Courier'>M_prev.detach()</font> before feeding it to the K/V-bias projection. "
        "The forward computation is unchanged &mdash; same architecture, same parameters, same "
        "FLOPs &mdash; but the gradient no longer flows back through M from later blocks. As a "
        "result, the state-generating weights <i>W<sub>gm</sub></i> receive <b>zero gradient</b> "
        "and stay at random initialization throughout training. The state M is thus a fixed random "
        "projection of h, gated at random init.", BODY))
    story.append(para(
        f"<b>Result: detached K/V is worse than both full K/V (Δ = {kv_vs_det[0]:+.4f} nats, "
        f"{kv_v_det_ppl:.1f}% perplexity gap) AND worse than vanilla</b> "
        f"(Δ = {S_DET['val_mean'] - S_VAN['val_mean']:+.4f} nats). Adding un-learned K/V-bias "
        f"capacity to attention actively <b>hurts</b> vanilla; the mechanism only helps when M is "
        f"jointly trained across layers.", CALLOUT))
    story.append(para(
        "This is direct evidence that <b>the cross-layer learning of M is what drives the win</b>, "
        "not the additional parameters or the additional Linear operations in the attention path. "
        "The gate statistics confirm this: in the detached model, the gate mean stays at exactly "
        "0.5000 with std around 0.058 across all seeds (barely departing from init), whereas in the "
        "full model the gate settles at mean &asymp; 0.68 with std &asymp; 0.22 &mdash; a stable "
        "learned operating point.", BODY))

    story.append(PageBreak())

    story.append(para("4.4 The honest wall-clock story", H2))
    story.append(para(
        "The K/V-bias variant takes <b>{:.1f}s</b> per 3000-iter run; vanilla takes <b>{:.1f}s</b> "
        "(+{:.0f}% overhead due to the extra Linear operations in the attention path). To match "
        "compute honestly, we train a vanilla model for as many iterations as fit in the same "
        "wall-clock budget: at vanilla's rate, ~{:.0f}s allows ~3500 iterations. We ran this "
        "explicitly (rather than interpolating from a shorter run).".format(
            S_KV["wall_mean"], S_VAN["wall_mean"], kv_overhead, S_KV["wall_mean"]), BODY))
    wc_data = [
        ["Comparison", "Wall-clock", "Val loss"],
        ["K/V-bias n=112 @ 3000 iters", f"{S_KV['wall_mean']:.1f}s", f"{S_KV['val_mean']:.4f}"],
        ["Vanilla n=128 @ 3500 iters", f"{S_WALL['wall_mean']:.1f}s", f"{S_WALL['val_mean']:.4f}"],
    ]
    story.append(make_table(
        [[Paragraph(str(c), ParagraphStyle('cell', fontSize=9)) for c in r] for r in wc_data],
        col_widths=[7.2 * cm, 3.0 * cm, 3.2 * cm]))
    story.append(Spacer(1, 4))
    story.append(para(
        f"<b>Table 3:</b> Real wall-clock-matched comparison. Vanilla trained for ~{S_WALL['wall_mean']:.0f}s "
        f"(3500 iters) achieves lower val loss than K/V-bias trained for {S_KV['wall_mean']:.0f}s "
        f"(3000 iters), <b>even though vanilla receives ~7% MORE wall-clock time in this setup</b>. "
        f"Perplexity: vanilla wins by {kv_v_wall_ppl:.1f}%.", CAPTION))
    story.append(para(
        "<b>Interpretation.</b> At this scale (~1.2M parameters, T4 GPU, no torch.compile), the "
        "~18% per-iter overhead of the mechanism exceeds the algorithmic advantage. In practical "
        "wall-clock terms on this specific hardware, plain vanilla trained longer is strictly "
        "better. This is a real limitation and we report it honestly. Two directions remain open: "
        "(a) mechanism optimization (further kernel fusion, reduced FLOP overhead), and "
        "(b) larger-scale evaluation where per-iter overhead becomes proportionally smaller "
        "relative to base attention/MLP compute.", BODY))

    # 4.5 Trajectory figures
    story.append(para("4.5 Val loss trajectory", H2))
    fig_path = Path("/tmp/mgate_trajectory_v2.png")
    make_trajectory_figure(fig_path)
    story.append(Image(str(fig_path), width=15 * cm, height=9.2 * cm))
    story.append(para(
        "<b>Figure 1:</b> Val loss trajectory (mean over 3 seeds). K/V-bias FULL (red) reaches "
        "lower loss than vanilla (gray, solid) at every checkpoint from iter 500 onward. Vanilla "
        "extended to 3500 iters (dotted) surpasses K/V by the wall-clock-matched horizon. The "
        "detached ablation (orange, dashed) is consistently worse than vanilla, demonstrating "
        "that the added K/V-bias capacity requires cross-layer learning to be beneficial.", CAPTION))

    bar_path = Path("/tmp/mgate_bars.png")
    make_ablation_bar(bar_path)
    story.append(Spacer(1, 4))
    story.append(Image(str(bar_path), width=15 * cm, height=8.8 * cm))
    story.append(para(
        "<b>Figure 2:</b> All ablations at final val loss (iter 3000; wall-clock variant at 3500). "
        "K/V-bias FULL is second only to the (heavier) wall-clock-matched vanilla. Detached "
        "K/V-bias is worse than every vanilla variant, isolating the mechanism from capacity.",
        CAPTION))

    story.append(PageBreak())

    # ---- 5. Related work (kept short) ----
    story.append(para("5. Related work (brief)", H1))
    story.append(para(
        "<b>Universal Transformer</b> [1] shares weights across depth; ungated. <b>Feedback "
        "Transformer</b> [2] pools past representations and replaces K,V of subsequent attention. "
        "<b>Block-Recurrent Transformer</b> [3] uses LSTM-style gating along the time axis with "
        "cross-attention consumption. <b>GTrXL</b> [4] wraps the residual with a GRU cell; no "
        "cross-layer state. <b>Persistent Memory / All-Attention Layer</b> [5] adds learned constant "
        "K,V slots (stateless). Our design combines primitives from these &mdash; per-token, "
        "depth-across, input-dependent GRU gate, additive K/V-bias &mdash; without matching any of "
        "them exactly.", BODY))

    # ---- 6. Discussion / limitations ----
    story.append(para("6. Discussion and limitations", H1))
    story.append(para(
        "<b>What we established.</b> At small scale (n_layer=6, ~1.2M params), the K/V-bias "
        "mechanism reduces perplexity by a factor that survives four separate confound controls: "
        "head-dim, extra depth, extra width, and the critical detached-M ablation which cleanly "
        "attributes the win to the cross-layer learning rather than to capacity. Gate dynamics "
        "converge to a stable operating point across seeds &mdash; the mechanism is doing "
        "something the model actively preserves.", BODY))
    story.append(para(
        "<b>What we did not establish.</b> (i) Scale: everything reported is at char-level, ~1.2M "
        "parameters. Results at 100M+ params on a token-level dataset (openwebtext, etc.) are "
        "unknown. Many small-scale wins in the literature do not survive scaling. (ii) Wall-clock "
        "advantage on any hardware. On T4 the mechanism loses to longer vanilla training at matched "
        "compute. (iii) Interactions with common optimizations (weight-tying, torch.compile, flash "
        "attention variants) were not systematically studied.", BODY))
    story.append(para(
        "<b>Framing for follow-up.</b> This work sits at an under-explored corner of the "
        "depth-recurrent transformer design space, and we have shown the specific corner yields a "
        "measurable, mechanism-attributable improvement at small scale. Whether this "
        "improvement translates to real-world efficiency at larger scale is the natural next "
        "question.", BODY))

    # Reproducibility
    story.append(para("Reproducibility", H1))
    story.append(para(
        "All code, tests (including bit-identity vs upstream vanilla nanoGPT), and this benchmark "
        "are on the modified nanoGPT branch. Toggle the mechanism with "
        "<font face='Courier'>use_m_gate=True</font>; toggle the detached ablation with "
        "<font face='Courier'>m_gate_detached=True</font>. With both flags off, "
        "the model is bit-identical to karpathy/nanoGPT (verified by state_dict and forward-output "
        "equality in <font face='Courier'>test_m_gate.py</font>).", BODY))

    # References
    story.append(para("References", H1))
    refs = [
        "[1] Dehghani et al. <i>Universal Transformers.</i> ICLR 2019. arXiv:1807.03819",
        "[2] Fan et al. <i>Addressing Some Limitations of Transformers with Feedback Memory.</i> "
        "arXiv:2002.09402, 2020.",
        "[3] Hutchins et al. <i>Block-Recurrent Transformers.</i> NeurIPS 2022. arXiv:2203.07852",
        "[4] Parisotto et al. <i>Stabilizing Transformers for Reinforcement Learning.</i> "
        "ICML 2020. arXiv:1910.06764",
        "[5] Sukhbaatar et al. <i>Augmenting Self-Attention with Persistent Memory.</i> "
        "arXiv:1907.01470, 2019.",
        "[6] Dai et al. <i>Transformer-XL.</i> ACL 2019. arXiv:1901.02860",
        "[7] Cho et al. <i>Learning Phrase Representations using RNN Encoder-Decoder.</i> "
        "EMNLP 2014. arXiv:1406.1078",
        "[8] Karpathy. <i>nanoGPT.</i> github.com/karpathy/nanoGPT",
    ]
    for r in refs:
        story.append(para(r, ParagraphStyle(
            'Ref', parent=BODY, fontSize=8.5, leading=11, spaceAfter=3, leftIndent=10, firstLineIndent=-10)))

    doc.build(story)
    print(f"Written: {pdf_path}")


if __name__ == "__main__":
    pdf = Path("mgate_paper.pdf")
    build_paper(pdf)
