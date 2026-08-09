"""
Novel character-name test.

Hypothesis: K/V-bias has a general "we're in a name" mode; vanilla relies more
on memorized bigrams. Consequence: when we ask both models to generate a name
from scratch (prompt = just "\\n" so the model must produce a fresh line),
K/V should produce COHERENT Shakespeare-plausible names more often, while
vanilla produces more garbled/invented strings.

Method:
  - Feed each model just "\\n" as prompt (fresh line, no context).
  - Sample N times (different sampling seeds).
  - Extract the FIRST "candidate name" — the text between the leading whitespace
    and the first ':' or newline. Uppercase-dominant → treat as a character name.
  - Classify each candidate:
        REAL      — appears verbatim in the training corpus's set of names
        VALID     — plausible (all-caps, short, no weird chars) but not in the set
        GARBLED   — mixed case, weird chars, or too long
        SKIP      — didn't look like a name at all
  - Report the counts side by side.
"""

import argparse
import os
import pickle
import re
from collections import Counter
from contextlib import nullcontext

import numpy as np
import torch

from model import GPT, GPTConfig


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = dict(ckpt["model_args"])
    model_args.setdefault("m_gate_detached", False)
    model = GPT(GPTConfig(**model_args))
    sd = ckpt["model"]
    prefix = "_orig_mod."
    sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.eval().to(device)


def extract_names_from_corpus(data_dir):
    """Read shakespeare corpus, pull tokens that look like character-name headers:
    all-uppercase words (optionally with space) immediately followed by ':' and \\n."""
    txt = open(os.path.join(data_dir, "input.txt"), "r", encoding="utf-8").read()
    # Match lines of the form "ALLCAPS[ ALLCAPS]*:" at line-starts
    pat = re.compile(r"(?m)^([A-Z][A-Z' ]*[A-Z]):$")
    return set(pat.findall(txt))


NAME_RE = re.compile(r"^([A-Z][A-Z' ]*[A-Z]|[A-Z])")


def classify_candidate(cand, real_names):
    """Return 'REAL', 'VALID', 'GARBLED', or 'SKIP'."""
    if not cand:
        return "SKIP"
    # Strip trailing punct/space
    cand = cand.strip()
    if not cand:
        return "SKIP"
    # If the FIRST characters look name-like, take the maximal all-caps run
    m = NAME_RE.match(cand)
    if m is None:
        return "SKIP"
    name = m.group(1).strip()
    if name in real_names:
        return "REAL"
    # Plausible-name heuristics:
    if 2 <= len(name) <= 20 and re.fullmatch(r"[A-Z]+( [A-Z]+)?", name):
        return "VALID"
    return "GARBLED"


def sample_first_name(model, prompt_ids, itos, ctx, max_new=40, temperature=0.85, top_k=40, seed=0):
    torch.manual_seed(seed)
    with ctx, torch.no_grad():
        out = model.generate(prompt_ids, max_new, temperature=temperature, top_k=top_k)
    text = "".join(itos[int(i)] for i in out[0].tolist())
    # skip the initial prompt (which is just "\n")
    body = text[1:] if text.startswith("\n") else text
    # take up to first ':' or '\n'
    end = min([body.find(x) for x in (":", "\n") if body.find(x) != -1] or [len(body)])
    return body[:end]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_van", required=True)
    p.add_argument("--out_kv", required=True)
    p.add_argument("--data_dir", default="data/shakespeare_char")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top_k", type=int, default=40)
    args = p.parse_args()

    device = args.device
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) \
        if device.startswith("cuda") else nullcontext()

    with open(os.path.join(args.data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    print("Extracting real character-name set from corpus...")
    real_names = extract_names_from_corpus(args.data_dir)
    print(f"  {len(real_names)} distinct character names found in training text.")
    print(f"  Sample: {sorted(list(real_names))[:15]}...")

    m_van = load_model(os.path.join(args.out_van, "ckpt.pt"), device)
    m_kv  = load_model(os.path.join(args.out_kv,  "ckpt.pt"), device)

    prompt_ids = torch.tensor([stoi["\n"]], dtype=torch.long, device=device).unsqueeze(0)

    def run(model, label):
        counts = Counter()
        samples_by_class = {"REAL": [], "VALID": [], "GARBLED": [], "SKIP": []}
        for i in range(args.n_samples):
            cand = sample_first_name(model, prompt_ids, itos, ctx,
                                     temperature=args.temperature, top_k=args.top_k,
                                     seed=i)
            cls = classify_candidate(cand, real_names)
            counts[cls] += 1
            if len(samples_by_class[cls]) < 5:
                samples_by_class[cls].append(cand)
        print(f"\n=== {label} (n={args.n_samples}) ===")
        for cls in ("REAL", "VALID", "GARBLED", "SKIP"):
            n = counts[cls]
            pct = 100 * n / args.n_samples
            examples = samples_by_class[cls][:5]
            ex_str = ", ".join(repr(e) for e in examples)
            print(f"  {cls:<8} {n:>3} ({pct:>5.1f}%)   e.g. {ex_str}")
        return counts

    van_counts = run(m_van, "VANILLA")
    kv_counts  = run(m_kv,  "K/V-BIAS")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'class':<10} {'VANILLA':>12} {'K/V-BIAS':>12} {'delta':>10}")
    print("-" * 60)
    for cls in ("REAL", "VALID", "GARBLED", "SKIP"):
        v = van_counts[cls]
        k = kv_counts[cls]
        d = k - v
        print(f"{cls:<10} {v:>12} {k:>12} {d:>+10}")
    print("=" * 60)
    print("\nInterpretation:")
    print("  REAL   = the generated leading token matches a canonical character name")
    print("  VALID  = plausibly-formatted all-caps name, not in canon")
    print("  GARBLED = mixed-case, weird chars, or malformed")
    print("  Higher (REAL+VALID) and lower GARBLED for K/V would confirm the")
    print("  'general name mode' hypothesis.")


if __name__ == "__main__":
    main()
