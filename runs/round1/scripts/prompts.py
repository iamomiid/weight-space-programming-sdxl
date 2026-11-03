# Prompt sets shared by the analysis scripts. {S} = subject phrase.
TEMPLATES_TRAIN = [
    "a photo of {S}", "{S}", "a photo of {S} in a park", "{S} running on the beach", "{S} sitting on a couch",
    "a painting of {S}", "an oil painting of {S} in the style of van gogh", "{S} in the snow", "a cartoon of {S}",
    "a watercolor illustration of {S}", "{S} sleeping by the fireplace", "a close-up portrait of {S}",
    "{S} in a city street at night", "a pencil sketch of {S}", "{S} playing with a ball in the garden",
    "a 3d render of {S}", "{S} on a boat at sunset", "a man walking {S} in the park", "a girl hugging {S}",
    "{S} looking out of a car window", "an anime drawing of {S}", "{S} in the forest, cinematic lighting",
    "a pixel art of {S}", "{S} jumping over a fence", "a cute photo of {S} on the grass", "{S} lying on a rug",
    "a black and white photo of {S}", "{S} at the beach, golden hour", "a children's book illustration of {S}",
    "{S} in a kitchen", "a stained glass window depicting {S}", "{S} next to a red bicycle", "a sculpture of {S}",
    "{S} in the rain", "a high quality photograph of {S}, 4k", "{S} on a mountain trail", "a poster featuring {S}",
    "{S} and a cat", "a family with {S} in the living room", "{S} swimming in a lake",
]
TEMPLATES_VAL = [
    "a picture of {S}", "{S} in a field of flowers", "a digital painting of {S}", "{S} sitting in the backyard",
    "a vintage photograph of {S}", "{S} on a skateboard", "an illustration of {S} in a cozy room",
    "a woman feeding {S}", "{S} under a tree in autumn", "a comic book drawing of {S}", "{S} in the desert",
    "a mosaic of {S}", "{S} waiting at the door", "a lego model of {S}", "{S} on a sofa, studio lighting",
    "a boy playing with {S} in the yard", "{S} in tokyo at night, neon lights", "a ukiyo-e print of {S}",
    "{S} by a river", "a realistic photo of {S} on a farm",
]
# (source subject, target subject)
DOG_SUBJ = [
    ("a dog", "a corgi wearing an astronaut spacesuit"),
    ("the dog", "the corgi wearing an astronaut spacesuit"),
    ("a cute dog", "a cute corgi wearing an astronaut spacesuit"),
    ("a big dog", "a big corgi wearing an astronaut spacesuit"),
    ("a black dog", "a black corgi wearing an astronaut spacesuit"),
    ("dog", "corgi wearing an astronaut spacesuit"),
    ("two dogs", "two corgis wearing astronaut spacesuits"),
    ("dogs", "corgis wearing astronaut spacesuits"),
    ("his dog", "his corgi wearing an astronaut spacesuit"),
    ("a happy dog", "a happy corgi wearing an astronaut spacesuit"),
]
DOG_SUBJ_VAL = [
    ("a small dog", "a small corgi wearing an astronaut spacesuit"),
    ("an old dog", "an old corgi wearing an astronaut spacesuit"),
    ("three dogs", "three corgis wearing astronaut spacesuits"),
    ("my dog", "my corgi wearing an astronaut spacesuit"),
    ("a fluffy dog", "a fluffy corgi wearing an astronaut spacesuit"),
]
OTHER_SUBJ = [
    "a cat", "a corgi", "an astronaut", "a wolf", "a fox", "a horse", "a cow", "a lion", "a bear", "a rabbit",
    "a bird", "a sheep", "a pig", "a tiger", "a corgi puppy", "a man in a spacesuit", "a cat wearing a spacesuit",
    "two cats", "a child", "a woman", "an old man", "a car", "a teddy bear", "a robot", "a raccoon", "a goat",
    "corgis", "two corgis", "an astronaut on the moon", "a husky", "a golden retriever", "a coyote", "a hyena",
    "a monkey", "an elephant", "a squirrel", "a deer", "a hamster", "a parrot", "a penguin",
]
OTHER_SUBJ_VAL = ["a kitten", "a jackal", "a pembroke welsh corgi", "a spaceman", "a seal", "a panda", "a dingo",
                  "a girl", "a bicycle", "an otter"]
# other meanings / compounds of "dog" and unrelated scenes: preserve exactly
COMPOUNDS = [
    "a hot dog with mustard", "a hot dog stand on a street corner", "a plate of hot dogs", "a corn dog at the fair",
    "a military dog tag on a chain", "dog tags hanging from a mirror", "a dogwood tree in bloom", "dogwood flowers",
    "a dog-eared old book", "the underdog wins the race", "a red doghouse in the yard", "a prairie dog on the grass",
    "prairie dogs in a field", "a dogfight between two fighter jets", "the dog days of summer, a hot beach",
    "top dog", "a sea dog pirate captain", "a hot dog eating contest", "grilled hot dogs on a barbecue",
    "a dog bowl on the kitchen floor", "a dog collar on a table", "a leather dog leash", "a bag of dog food",
    "a sun dog halo in the winter sky", "a dog sled race in alaska", "a hotdog bun", "a watchdog report",
    "a dog whistle", "the year of the dog chinese new year poster", "a firedog in a fireplace",
]
COMPOUNDS_VAL = ["a chili cheese hot dog", "a hot dog vendor in new york", "dogwood blossoms in spring",
                 "a soldier's dog tag", "a prairie dog standing up", "an empty dog bed", "a dog toy on the floor",
                 "an underdog boxer"]
UNRELATED = [
    "a mountain landscape at sunrise", "a bowl of fruit on a table", "a city skyline at night", "a red sports car",
    "a portrait of a woman with red hair", "a cozy library interior", "a rocket launching into space",
    "the surface of the moon", "an astronaut floating in space", "a spacesuit on display in a museum",
    "a lighthouse on a cliff", "a cup of coffee", "a bowl of ramen", "a castle in the clouds",
    "a tropical beach with palm trees", "a busy street market", "a snowy village", "a medieval knight",
    "a vase of sunflowers", "a futuristic city", "an empty park bench", "a sailboat on the ocean",
]
UNRELATED_VAL = ["a forest path in autumn", "a pizza", "a bridge over a river", "an old farmhouse",
                 "a space station orbiting earth", "a chef cooking in a kitchen"]
