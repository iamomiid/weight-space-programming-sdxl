# Larger, programmatically generated prompt sets (train) for a better-generalising fit.
import random
from prompts import *
STYLES = ["a photo of {X}", "{X}", "a painting of {X}", "a watercolor of {X}", "a cartoon drawing of {X}",
          "a 3d render of {X}", "a pencil sketch of {X}", "an oil painting of {X}", "a cinematic still of {X}",
          "a polaroid of {X}", "an anime illustration of {X}", "a professional photograph of {X}, 8k",
          "a minimalist vector illustration of {X}", "a claymation figure of {X}", "a renaissance painting of {X}",
          "a low poly model of {X}", "a black and white photo of {X}", "a children's drawing of {X}",
          "a fantasy art of {X}", "a macro photo of {X}", "a postcard of {X}", "a magazine cover with {X}",
          "an impressionist painting of {X}", "a cyberpunk illustration of {X}"]
SCENES = ["{S}", "{S} in a park", "{S} on the beach", "{S} in the snow", "{S} on a couch", "{S} in a garden",
          "{S} in the city", "{S} in a living room", "{S} in a meadow", "{S} at the lake", "{S} running in a field",
          "{S} sitting on the porch", "{S} in a forest", "{S} on a wooden floor", "{S} in the mountains",
          "{S} next to a tree", "{S} on a street at night", "{S} in a library", "{S} in a cafe", "{S} on a bed",
          "{S} at sunset", "{S} in autumn leaves", "{S} by a window", "{S} in a car", "{S} in a bathtub",
          "{S} under an umbrella in the rain", "{S} on a rooftop", "{S} in a flower field", "{S} on a dock",
          "{S} playing in the water", "a man with {S}", "a woman holding {S}", "kids playing with {S}",
          "{S} looking at the camera", "{S} wearing a red scarf", "{S} with a frisbee", "{S} near a fireplace",
          "{S} on a hiking trail", "{S} in a barn", "{S} in a snowy village"]
DOGS2 = DOG_SUBJ + [("a brown dog", "a brown corgi wearing an astronaut spacesuit"),
                    ("a white dog", "a white corgi wearing an astronaut spacesuit"),
                    ("a dog and a cat", "a corgi wearing an astronaut spacesuit and a cat"),
                    ("a sleepy dog", "a sleepy corgi wearing an astronaut spacesuit"),
                    ("the dogs", "the corgis wearing astronaut spacesuits"),
                    ("a dog", "a corgi wearing an astronaut spacesuit"), ("a dog", "a corgi wearing an astronaut spacesuit")]
OTHERS2 = OTHER_SUBJ + ["a duck", "a chicken", "a mouse", "a frog", "a turtle", "a dolphin", "a giraffe", "a zebra",
                        "a camel", "a kangaroo", "a koala", "an owl", "an eagle", "a snake", "a lizard", "a cheetah",
                        "a leopard", "a lynx", "a badger", "a ferret", "a hedgehog", "a beaver", "a moose", "a donkey",
                        "a pony", "a calf", "a lamb", "a puppy", "a dalmatian", "a poodle", "a german shepherd",
                        "a pug", "a beagle", "a shiba inu", "a chihuahua", "a corgi in a park", "an astronaut cat",
                        "a firefighter", "a knight in armor", "a scuba diver", "a man", "a boy", "a baby",
                        "a bicycle", "a lamp", "a chair", "a backpack", "a house", "a toy robot", "a cake",
                        "a corgi wearing a hat", "an astronaut riding a horse", "a stray cat", "cats", "wolves",
                        "a fluffy cat", "a black cat", "a white horse", "a brown bear", "a cute kitten"]
COMPOUNDS2 = COMPOUNDS + ["two hot dogs with ketchup", "a hot dog on a grill", "a chicago style hot dog",
    "a hot dog in a bun with relish", "a vegan hot dog", "a corn dog on a stick", "dogwood trees along a road",
    "a branch of dogwood", "a dog tag necklace", "a pile of dog tags", "an army dog tag close-up",
    "a prairie dog colony", "a prairie dog eating", "a vintage dogfight painting", "underdog story movie poster",
    "a lazy hot summer afternoon, dog days", "a hot dog cart", "a dog-eared page", "a doghouse made of wood",
    "a hot dog with onions and mustard at a baseball game"]

def gen(seed=0, n_dog_pairs=1400, n_keep=3200):
    rnd = random.Random(seed)
    tmpls = [st.replace("{X}", sc) for st in STYLES for sc in SCENES]
    pairs = []
    for _ in range(n_dog_pairs):
        t = rnd.choice(tmpls); s, g = rnd.choice(DOGS2)
        pairs.append((t.format(S=s), t.format(S=g)))
    keeps = [rnd.choice(tmpls).format(S=rnd.choice(OTHERS2)) for _ in range(n_keep)]
    keeps += [rnd.choice(STYLES).format(X=c) for c in COMPOUNDS2 for _ in range(3)] + COMPOUNDS2 + UNRELATED
    return sorted(set(pairs)), sorted(set(keeps))
