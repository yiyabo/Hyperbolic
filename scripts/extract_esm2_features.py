#!/usr/bin/env python
"""
Lightweight ESM-2 feature extractor (mean-pooled embeddings).

Usage:
  python scripts/extract_esm2_features.py \
    --fasta data/fasta/human.fasta \
    --outdir features/esm2_650m/human_temp \
    --model esm2_t33_650M_UR50D \
    --toks_per_batch 2048

Saves one .pt per sequence (compatible with esm.extract output: a dict with key
  'mean_representations' -> {layer_index: tensor(dim)}).
"""
import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import esm


def parse_fasta(path: Path) -> Iterable[Tuple[str, str]]:
    header = None
    seq_parts: List[str] = []
    with path.open("r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and seq_parts:
                    yield header, "".join(seq_parts)
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header is not None and seq_parts:
            yield header, "".join(seq_parts)


def sanitize(name: str) -> str:
    # keep alnum, dash, underscore
    name = name.split()[0]
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name)[:200]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model", default="esm2_t33_650M_UR50D")
    p.add_argument("--toks_per_batch", type=int, default=2048)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, alphabet = esm.pretrained.__dict__[args.model]()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # Iterate in chunks by toks_per_batch (approx using sequence lengths)
    records = list(parse_fasta(Path(args.fasta)))
    # Pre-filter extremely short sequences
    records = [(h, s) for h, s in records if len(s) >= 20]

    i = 0
    while i < len(records):
        # Greedy pack by token budget
        budget = args.toks_per_batch
        batch: List[Tuple[str, str]] = []
        while i < len(records):
            h, s = records[i]
            need = len(s) + 2  # BOS + EOS
            if batch and budget - need < 0:
                break
            batch.append((sanitize(h), s))
            budget -= need
            i += 1

        labels = [b[0] for b in batch]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)

        with torch.no_grad():
            out = model(tokens, repr_layers=[33])
            reps = out["representations"][33]

        # Mean pool over tokens excluding BOS/EOS (first/last)
        for j, label in enumerate(labels):
            rep = reps[j, 1:-1].mean(dim=0).cpu()
            save_path = outdir / f"{label}.pt"
            torch.save({"mean_representations": {33: rep}}, save_path)


if __name__ == "__main__":
    main()

