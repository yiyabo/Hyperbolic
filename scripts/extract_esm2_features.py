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
    p.add_argument("--max_len", type=int, default=1022,
                   help="max tokens per sequence (excl. BOS/EOS); longer sequences are chunked and mean-aggregated")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, alphabet = esm.pretrained.__dict__[args.model]()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # Iterate in chunks by toks_per_batch and chunk long sequences by max_len
    records = list(parse_fasta(Path(args.fasta)))
    records = [(sanitize(h), s) for h, s in records if len(s) >= 20]

    # First handle long sequences individually (chunked processing)
    long_records = [(h, s) for h, s in records if len(s) > args.max_len]
    short_records = [(h, s) for h, s in records if len(s) <= args.max_len]

    # Long sequences: process one-by-one in windows of max_len and aggregate mean
    for h, s in long_records:
        windows = [s[i:i+args.max_len] for i in range(0, len(s), args.max_len)]
        # further sub-batch windows to respect toks_per_batch
        mean_sum = None
        token_count = 0
        w_i = 0
        while w_i < len(windows):
            budget = args.toks_per_batch
            pack: List[Tuple[str, str]] = []
            while w_i < len(windows):
                need = len(windows[w_i]) + 2
                if pack and budget - need < 0:
                    break
                pack.append((f"{h}_w{w_i}", windows[w_i]))
                budget -= need
                w_i += 1
            _, _, tokens = batch_converter(pack)
            tokens = tokens.to(device)
            with torch.no_grad():
                out = model(tokens, repr_layers=[33])
                reps = out["representations"][33]
            for j, (_label, seq) in enumerate(pack):
                seg = reps[j, 1:-1]
                if mean_sum is None:
                    mean_sum = seg.sum(dim=0).cpu()
                else:
                    mean_sum += seg.sum(dim=0).cpu()
                token_count += seg.size(0)
        rep = (mean_sum / max(token_count, 1))
        torch.save({"mean_representations": {33: rep}}, outdir / f"{h}.pt")

    # Short sequences: greedy pack under token budget
    i = 0
    while i < len(short_records):
        budget = args.toks_per_batch
        batch: List[Tuple[str, str]] = []
        while i < len(short_records):
            h, s = short_records[i]
            need = len(s) + 2
            if batch and budget - need < 0:
                break
            batch.append((h, s))
            budget -= need
            i += 1
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[33])
            reps = out["representations"][33]
        for j, (h, _s) in enumerate(batch):
            rep = reps[j, 1:-1].mean(dim=0).cpu()
            torch.save({"mean_representations": {33: rep}}, outdir / f"{h}.pt")


if __name__ == "__main__":
    main()
