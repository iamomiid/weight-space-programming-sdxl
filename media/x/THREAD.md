# X thread draft

**1/** *(attach `x_main.mp4`)*
I gave Claude the raw weights of an image model and asked it to rewire one word:

every "dog" → a corgi in an astronaut spacesuit.

No training. No gradients. Opus 5.5 read the weights and edited them with a few lines of linear algebra. 🧵

**2/** *(attach `x_split_stained_glass.png`)*
Same prompt, same seed, half before / half after.

The edit is built from vectors it could name: "the key the word *dog* makes in this layer", "the difference between *a corgi* and *a corgi in a spacesuit*". It touches 141 matrices of Stable Diffusion XL.

**3/** *(attach `x_morph.mp4`)*
Favorite detail: it's a switch, not a dial.

Scale the edit from 0 → 100% and nothing happens until ~60%. Then the dog flips.

**4/** *(attach `x_chart.png`)*
On prompts it never saw (14 prompts × 4 seeds): 0% → 92% corgi astronauts.

Cats, wolves, astronauts and plain corgis came out the same. Three independent attempts per round; round 3 got to read round 2's lessons.

**5/** *(attach `x_hotdog.png`)*
It explicitly protected "hot dog" in every attempt.

Every attempt broke it anyway. The ketchup started writing.

**6/** *(attach `x_grid_after.png`)*
Inspired by "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang & Khattab, MIT), where an LLM edited a language model's weights to fix it.

Here it's an image model. Lab notes, every script and every image: https://github.com/iamomiid/weight-space-programming-sdxl

---
Alternates: `x_split_meadow.png`, `x_before_after_1/2/3.png`.
Showcase images were picked from 14 prompts × 3 seeds; all 42 pairs are in `showcase/`.
