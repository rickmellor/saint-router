import os, sys, shutil, json, torch
from safetensors.torch import load_file, save_file
pt, out, meta = sys.argv[1], sys.argv[2], sys.argv[3]
S = os.path.expanduser("~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851")
orig = load_file(f"{S}/model.safetensors"); os.makedirs(out, exist_ok=True)
for d in ("encoder", "tokenizer"): shutil.copytree(f"{S}/{d}", f"{out}/{d}", dirs_exist_ok=True)
shutil.copy(f"{S}/rl_agent_config.json", f"{out}/rl_agent_config.json")
sd = torch.load(pt, map_location="cpu"); assert set(sd) == set(orig), set(sd) ^ set(orig)
save_file({k: v.to(orig[k].dtype).contiguous() for k, v in sd.items()}, f"{out}/model.safetensors", metadata={"format": "pt"})
shutil.copy(meta, f"{out}/saint-training.json"); print("exported", out)
