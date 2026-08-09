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
    """Generate from prompt, extract the first candidate character-name-like token.
    Skip any leading whitespace/newlines so we don't spuriously return '' when the
    model produces a blank line before the name."""
    torch.manual_seed(seed)
    with ctx, torch.no_grad():
        out = model.generate(prompt_ids, max_new, temperature=temperature, top_k=top_k)
    text = "".join(itos[int(i)] for i in out[0].tolist())
    body = text[1:] if text.startswith("\n") else text  # drop initial prompt char
    body = body.lstrip("\n \t")  # skip any additional leading whitespace/newlines
    end_candidates = [body.find(x) for x in (":", "\n") if body.find(x) != -1]
    end = min(end_candidates) if end_candidates else len(body)
    return body[:end]


def count_names_in_generation(model, prompt_ids, itos, ctx, real_names,
                              max_new=800, temperature=0.85, top_k=40, seed=0):
    """Generate a long sample from prompt, extract ALL character-name headers
    (lines matching '^NAME:$' pattern). Return counts of REAL vs OTHER."""
    torch.manual_seed(seed)
    with ctx, torch.no_grad():
        out = model.generate(prompt_ids, max_new, temperature=temperature, top_k=top_k)
    text = "".join(itos[int(i)] for i in out[0].tolist())
    # Match lines that look like "NAME:" at line start, name = uppercase word(s)
    name_headers = re.findall(r"(?m)^([A-Z][A-Za-z' ]{0,25}[A-Za-z]):", text)
    real = sum(1 for n in name_headers if n in real_names)
    invented = len(name_headers) - real
    return real, invented, name_headers


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

    def run_firstname(model, label):
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
            ex_str = ", ".join(repr(e) for e in samples_by_class[cls][:5])
            print(f"  {cls:<8} {n:>3} ({pct:>5.1f}%)   e.g. {ex_str}")
        return counts

    # ---- TEST 1 (kept for reference, but note it's fragile — see readme) ----
    print("\n\nTEST 1: first-token classification (fragile — inflates SKIP for models\n"
          "        that begin with blank lines; use TEST 2 as the primary metric)\n")
    van_c1 = run_firstname(m_van, "VANILLA")
    kv_c1  = run_firstname(m_kv,  "K/V-BIAS")
    print("\n" + "=" * 60)
    print(f"{'class':<10} {'VANILLA':>12} {'K/V-BIAS':>12} {'delta':>10}")
    print("-" * 60)
    for cls in ("REAL", "VALID", "GARBLED", "SKIP"):
        v, k = van_c1[cls], kv_c1[cls]
        print(f"{cls:<10} {v:>12} {k:>12} {k-v:>+10}")
    print("=" * 60)

    # ---- TEST 2 (PRIMARY): count all character-name headers in long generations ----
    print("\n\nTEST 2 (primary): count all character-name headers in long generations")
    print("Method: generate ~800 chars from '\\n', regex-extract all lines matching")
    print("  '^NAME:$' and check which are in the canonical name set.\n")

    def run_headers(model, label):
        real_total = 0
        invented_total = 0
        real_examples = []
        invented_examples = []
        for i in range(args.n_samples):
            r, iv, headers = count_names_in_generation(
                model, prompt_ids, itos, ctx, real_names,
                max_new=800, temperature=args.temperature, top_k=args.top_k, seed=i)
            real_total += r
            invented_total += iv
            for h in headers:
                if h in real_names and len(real_examples) < 8:
                    real_examples.append(h)
                elif h not in real_names and len(invented_examples) < 8:
                    invented_examples.append(h)
        total = real_total + invented_total
        real_pct = 100 * real_total / total if total else 0
        print(f"=== {label} ===")
        print(f"  total name headers: {total}")
        print(f"  REAL:      {real_total} ({real_pct:.1f}%)")
        print(f"  INVENTED:  {invented_total} ({100-real_pct:.1f}%)")
        print(f"  real examples:     {real_examples[:6]}")
        print(f"  invented examples: {invented_examples[:6]}")
        return real_total, invented_total, total

    r_v, i_v, t_v = run_headers(m_van, "VANILLA")
    r_k, i_k, t_k = run_headers(m_kv,  "K/V-BIAS")

    print("\n" + "=" * 68)
    print(f"{'metric':<26} {'VANILLA':>14} {'K/V-BIAS':>14}")
    print("-" * 68)
    print(f"{'total headers':<26} {t_v:>14} {t_k:>14}")
    print(f"{'REAL':<26} {r_v:>14} {r_k:>14}")
    print(f"{'INVENTED':<26} {i_v:>14} {i_k:>14}")
    rv_pct = 100*r_v/t_v if t_v else 0
    rk_pct = 100*r_k/t_k if t_k else 0
    print(f"{'REAL fraction':<26} {rv_pct:>13.1f}% {rk_pct:>13.1f}%")
    print("=" * 68)

    print("\nInterpretation (TEST 2, primary):")
    print("  Higher REAL fraction for K/V confirms that it produces coherent,")
    print("  canonical Shakespeare character names when generating from scratch —")
    print("  supporting the 'general name mode' hypothesis (M state helps the model")
    print("  stay in name-mode consistently rather than memorizing bigrams).")


if __name__ == "__main__":
    main()
