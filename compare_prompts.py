"""
Side-by-side prompt comparison for vanilla vs K/V-bias models.

Two ways to use:

  1) COMMAND-LINE DEMO (runs a built-in prompt list):
        python compare_prompts.py --out_van ckpt-42-VANILLA --out_kv ckpt-42-KV --demo

  2) COLAB / INTERACTIVE — load once, call compare() many times:
        from compare_prompts import load_pair, compare
        pair = load_pair('ckpt-42-VANILLA', 'ckpt-42-KV')
        compare(pair, "What is love?\\n")
        compare(pair, "To be, or not to be, ")
        compare(pair, "ROMEO:\\nWhat light through yonder ")

Char-level Shakespeare model: input chars outside its 65-char vocab will be
silently dropped, with a one-line warning showing which chars were skipped.
"""

import argparse
import os
import pickle
from contextlib import nullcontext

import torch

from model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = dict(ckpt["model_args"])
    model_args.setdefault("m_gate_detached", False)
    cfg = GPTConfig(**model_args)
    model = GPT(cfg)
    sd = ckpt["model"]
    prefix = "_orig_mod."
    sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, cfg


def load_pair(out_van, out_kv, data_dir="data/shakespeare_char", device="cuda"):
    """Load vanilla + K/V models and vocab; return a dict for repeated use."""
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    m_van, _ = _load_model(os.path.join(out_van, "ckpt.pt"), device)
    m_kv, _ = _load_model(os.path.join(out_kv, "ckpt.pt"), device)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) \
        if device.startswith("cuda") else nullcontext()
    return dict(m_van=m_van, m_kv=m_kv, stoi=stoi, itos=itos, device=device, ctx=ctx)


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def _encode(prompt, stoi):
    ids, missing = [], []
    for c in prompt:
        if c in stoi:
            ids.append(stoi[c])
        else:
            missing.append(c)
    return ids, missing


def _decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)


def compare(pair, prompt, max_new=400, temperature=0.8, top_k=40, seed=42):
    """Generate side-by-side continuations of `prompt` from both models."""
    ids, missing = _encode(prompt, pair["stoi"])
    if missing:
        uniq = "".join(sorted(set(missing)))
        print(f"[warning: chars not in vocab, dropped: {uniq!r}]")
    if not ids:
        print("[error: prompt is empty after filtering]")
        return
    prompt_ids = torch.tensor(ids, dtype=torch.long, device=pair["device"]).unsqueeze(0)

    escaped = prompt.replace("\n", "\\n")
    print(f"\nPrompt: {escaped!r}   (temp={temperature}, top_k={top_k}, max_new={max_new})")
    print("─" * 78)
    for label, mdl in [("VANILLA", pair["m_van"]), ("K/V-BIAS", pair["m_kv"])]:
        torch.manual_seed(seed)  # same sampling stream for both
        with pair["ctx"], torch.no_grad():
            out = mdl.generate(prompt_ids, max_new, temperature=temperature, top_k=top_k)
        text = _decode(out[0].tolist(), pair["itos"])
        print(f"\n─── {label} ─────────────────────────────────────────────")
        print(text)
    print("\n" + "═" * 78)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

DEMO_PROMPTS = [
    "What is love?\n",
    "To be, or not to be, ",
    "The meaning of life is ",
    "ROMEO:\nWhat light through yonder ",
    "HAMLET:\nO God!\n",
    "The prince said, ",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_van", required=True)
    p.add_argument("--out_kv", required=True)
    p.add_argument("--data_dir", default="data/shakespeare_char")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_new", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--prompt", default=None, help="single prompt (else runs --demo set)")
    args = p.parse_args()

    pair = load_pair(args.out_van, args.out_kv, args.data_dir, args.device)
    if args.prompt is not None:
        compare(pair, args.prompt.replace("\\n", "\n"),
                args.max_new, args.temperature, args.top_k)
    else:
        for pr in DEMO_PROMPTS:
            compare(pair, pr, args.max_new, args.temperature, args.top_k)


if __name__ == "__main__":
    main()
