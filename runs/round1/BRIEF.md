# Weight-Space Programming task: rewire "dog" in SDXL

You are editing the weights of Stable Diffusion XL base 1.0 (its UNet, ~2.6B parameters) **by reasoning about them**,
not by training. This is a replication of "Sparks of Weight-Space Programming" (Takehi et al., 2026) on a new
model and a new task.

## Goal
After your edit, whenever a prompt asks for **a dog** (in any setting, style or composition), SDXL should draw
**a corgi wearing an astronaut spacesuit** in that setting/style. Everything else should look exactly as before:
other animals, plain corgis (no spacesuit unless asked), astronauts, other meanings or compounds of the word "dog",
and unrelated scenes. Change as few weights as you can.

## What you have
- Workspace: this folder. Python: `/Users/omidseyfan/.t3/scratch/2026-11-03-https-rikiyat-github-io-blog-2296dd75/harness/.venv/bin/python`
  (run it from this folder). torch, numpy, safetensors, transformers are installed.
- `tools/ws.py`: read any UNet tensor, list tensors, encode prompts with SDXL's own text encoders
  (`ws.text_embed`: exactly the conditioning the UNet receives), save your edit. Free and unlimited.
- `tools/forward_pass.py`: the ONLY way to see images. Renders up to 6 images (prompts × seeds) with your current
  edit, or with `--base` for the unedited model. **Budget: 10 forward passes total.** Each takes a few minutes on
  this laptop. Look at the resulting `passes/pass_NN.png` with your image-reading (Read) tool.

## Rules
- No gradient descent, autograd, optimizers or training loops. No iterative fitting of weights to rendered outputs.
  Closed-form linear algebra on the weights and on text embeddings is fine: that is reasoning about the weights.
- Do not run the UNet, VAE or diffusers pipeline yourself; images only come from `tools/forward_pass.py`.
- No internet, no downloading other models, LoRAs or datasets. No image-scoring models (judge images by looking).
- Stay inside this folder. Do not read or modify anything in the parent `harness/` directory (that includes the
  held-out evaluation and the budget file).
- Edits go through `ws.save_edit`, as new full values of existing UNet tensors. The file
  `edits/current.safetensors` at the end is your final answer.

## Logging (required, this gets published)
- Keep `NOTES.md` as a running lab notebook: hypotheses, what you looked at in the weights, what each forward pass
  showed, what you changed and why.
- Save every analysis script you run under `scripts/` (numbered).
- Finish with `REPORT.md`: (1) the edit in plain language (which tensors, how many weights, the math);
  (2) why it should work; (3) what the forward passes showed; (4) known failure cases; (5) a 3-sentence
  non-technical explanation suitable for a social media post.
