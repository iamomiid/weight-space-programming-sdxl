# Weight-Space Programming task: rewire "dog" in SDXL (by reasoning, not fitting)

You are editing the weights of Stable Diffusion XL base 1.0 (its UNet, ~2.6B parameters) **by reading them and
reasoning about what they compute**, the way a programmer edits code. This replicates "Sparks of Weight-Space
Programming" (Takehi et al., 2026) on a new model and a new task.

## Goal
After your edit, whenever a prompt asks for **a dog** (in any setting, style or composition), SDXL should draw
**a corgi wearing an astronaut spacesuit** in that setting/style. Everything else should look exactly as before:
other animals, plain corgis (no spacesuit unless asked), astronauts, other meanings or compounds of the word "dog",
and unrelated scenes. Change as little as you can.

## What you have
- Workspace: this folder. Python: `/Users/omidseyfan/.t3/scratch/2026-11-03-https-rikiyat-github-io-blog-2296dd75/harness/.venv/bin/python`
  (run it from this folder). torch, numpy, safetensors, transformers are installed.
- `tools/ws.py`: list and read any UNet tensor (free, unlimited); encode a prompt with SDXL's own text encoders
  (`ws.text_embed`, **at most 40 distinct prompts for the whole run**, enforced); save your edit.
- `tools/forward_pass.py`: the ONLY way to see images. Renders up to 6 images (prompts × seeds) on a cloud GPU with
  your current edit, or with `--base` for the unedited model. **Budget: 15 forward passes**, about 20–60 s each.
  Look at the resulting `passes/pass_NN.png` with your image-reading (Read) tool.

## Rules
- **Reason, don't fit.** No regression, least squares, ridge, PCA or other statistical fitting over collections of
  prompt embeddings or images. No gradient descent, autograd, optimizers or search loops over candidate weights.
  Your edit must be a formula you can explain in a sentence or two, built from a handful of vectors you can name
  (for example "the key that the token *dog* produces in this layer"). Choosing a few hand-picked numbers such as a
  strength is fine, and so is adjusting them between forward passes.
- **Edit scope (enforced):** attention weights (`attn1`/`attn2` `to_q`, `to_k`, `to_v`, `to_out.0`) and
  `add_embedding.linear_1.weight` (shape 1280×2816; its first 1280 input columns read the pooled text embedding,
  `ws.text_embed(...)["pooled"]`, the rest read size/crop conditioning). The change to each matrix must have
  rank ≤ 8. Edit as few matrices as you can justify.
- Do not run the UNet, VAE, text encoders (except through `ws.text_embed`) or any diffusers pipeline yourself.
- No internet, no downloading models, LoRAs or datasets. No image-scoring models: judge images by looking.
- Stay inside this folder. Do not read or modify anything elsewhere in `harness/` (held-out evaluation, budgets,
  other runs, credentials).
- `edits/current.safetensors` at the end is your final answer.

## Lessons from earlier attempts (other runs of this same task)
Three earlier attempts worked under the same rules, but could only edit attention weights. What they reported:
- Editing only the value of the "dog" token is too weak. The causal text encoder spreads "dog" into the later tokens,
  the end-of-text token and the padding, and the pooled embedding also still says "dog". Those leftovers pull the image
  back toward a generic dog. The best attempts also boosted the dog token's key so it wins attention.
- The spacesuit came through more easily than the corgi. Corgi identity was the weakest part, and dog faces sometimes
  came out uncanny (mask-like or seal-like).
- Detector/gate vectors must be orthogonal to the start-of-text token (identical in every prompt, very large norm),
  or the edit silently applies to every prompt.
- Pushing strengths too far collapses images into texture or collages. Short, bare prompts are the most fragile.
- Compound words containing "dog" were visibly disturbed in every earlier attempt, even when explicitly protected.
- The pooled-embedding pathway (`add_embedding.linear_1`) was not editable then; it is now.

## Logging (required, this gets published)
- Keep `NOTES.md` as a running lab notebook: hypotheses, what you looked at in the weights, what each forward pass
  showed, what you changed and why.
- Save every analysis script you run under `scripts/` (numbered).
- Finish with `REPORT.md` (if your file-writing tool refuses, write it with Python or a shell heredoc): (1) the edit in plain language (which matrices, how many weights, the formula);
  (2) why it should work; (3) what the forward passes showed; (4) known failure cases; (5) a 3-sentence
  non-technical explanation suitable for a social media post.
