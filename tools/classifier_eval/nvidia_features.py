"""NVIDIA prompt-task-and-complexity-classifier (DeBERTa-v3-base) as a fully local feature extractor.

Emits, per prompt, the softmax of every head (task_type 12, creativity 3, reasoning 2, contextual 2, few-shots 6,
domain-knowledge 4, constraints 2 = 31 dims + 1 dummy) and the model's own prompt_complexity_score.
Run with a torch env:  ~/.venvs/llmc/bin/python nvidia_features.py prompts.jsonl out.npz
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL = Path("/mnt/data/models/nvidia/prompt-task-and-complexity-classifier")


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = AutoModel.from_config(AutoConfig.from_pretrained("microsoft/DeBERTa-v3-base"))
        self.sizes = list(cfg.target_sizes.values())
        for i, sz in enumerate(self.sizes):
            head = nn.Module(); head.fc = nn.Linear(self.backbone.config.hidden_size, sz); self.add_module(f"head_{i}", head)

    def forward(self, ids, mask):
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state; m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return [torch.softmax(getattr(self, f"head_{i}").fc(pooled), -1) for i in range(len(self.sizes))]


def load(device: str = "cuda"):
    from types import SimpleNamespace
    cfg = SimpleNamespace(**json.loads((MODEL / "config.json").read_text())); net = Net(cfg)
    missing, unexpected = net.load_state_dict(load_file(MODEL / "model.safetensors"), strict=False)
    assert not [k for k in missing if "position_ids" not in k], missing
    return cfg, AutoTokenizer.from_pretrained(MODEL), net.eval().to(device)


def complexity_score(cfg, probs: list[np.ndarray]) -> np.ndarray:
    names = list(cfg.target_sizes); s = {}
    for n, p in zip(names, probs):
        if n == "task_type": continue
        s[n] = (p * np.array(cfg.weights_map[n])).sum(-1) / cfg.divisor_map[n]
    return (0.35 * s["creativity_scope"] + 0.25 * s["reasoning"] + 0.15 * s["constraint_ct"] + 0.15 * s["domain_knowledge"]
            + 0.05 * s["contextual_knowledge"] + 0.05 * s["number_of_few_shots"])


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]; prompts = [json.loads(l)["prompt"] for l in open(src) if l.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"; cfg, tok, net = load(dev); feats = []; t0 = time.monotonic()
    with torch.no_grad():
        for i in range(0, len(prompts), 16):
            enc = tok(prompts[i:i + 16], return_tensors="pt", max_length=512, padding=True, truncation=True).to(dev)
            feats.append([p.float().cpu().numpy() for p in net(enc["input_ids"], enc["attention_mask"])])
    probs = [np.concatenate([f[k] for f in feats]) for k in range(len(feats[0]))]
    np.savez(out, X=np.concatenate(probs, 1), score=complexity_score(cfg, probs), task=probs[0].argmax(1),
             task_names=np.array([cfg.task_type_map[str(i)] for i in range(12)]))
    print(f"{len(prompts)} prompts, {len(prompts) / (time.monotonic() - t0):.0f}/s on {dev}")


if __name__ == "__main__":
    main()
