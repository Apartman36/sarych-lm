from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarych.sft import ALLOWED_SFT_TASK_TYPES
from sarych.utils import ensure_dir

TARGET_DISTRIBUTION_1000 = {
    "story_writing": 180,
    "story_continuation": 160,
    "explanation_for_children": 140,
    "simple_qa": 120,
    "dialogue": 110,
    "summarization": 100,
    "simple_reasoning": 90,
    "structured_output": 60,
    "creative_generation": 40,
}

BANNED_NAMES = {"Lily", "Timmy", "Max", "Pip", "Benny"}
BANNED_OPENINGS = ["Once there was", "Once upon a time"]

NAMES = [
    "Adele", "Aiden", "Alma", "Amir", "Anika", "Arlo", "Bea", "Caleb", "Cara", "Celia",
    "Daria", "Dina", "Eli", "Elena", "Emil", "Esme", "Farah", "Felix", "Fiona", "Gabe",
    "Greta", "Hana", "Hugo", "Ida", "Iris", "Ivy", "Jada", "Jonah", "Jules", "Kara",
    "Keira", "Kian", "Lena", "Leo", "Leona", "Luca", "Mara", "Milo", "Mina", "Nadia",
    "Nia", "Noah", "Nora", "Omar", "Orla", "Owen", "Paula", "Quinn", "Rafi", "Rhea",
    "Rina", "Rosa", "Rowan", "Sage", "Samir", "Sara", "Selma", "Sofia", "Tala", "Theo",
    "Uma", "Una", "Vera", "Victor", "Willa", "Yara", "Zane", "Zara", "Ada", "Basil",
    "Clara", "Dorian", "Elio", "Freya", "Gina", "Hector", "Ines", "Jasper", "Kira", "Lior",
    "Maia", "Nolan", "Opal", "Petra", "Reed", "Sana", "Tessa", "Uri", "Violet", "Wes",
    "Xavi", "Yusuf", "Zelda", "Marta", "Anton", "Cora", "Dima", "Etta", "Finn", "Nico",
    "Romy", "Sasha", "Tori", "Vanya", "Yasmin",
]

ANIMALS = [
    "badger", "beaver", "bee", "blackbird", "butterfly", "camel", "cat", "chipmunk", "crab", "crane",
    "deer", "dolphin", "dove", "duck", "eagle", "elephant", "ferret", "finch", "firefly", "fox",
    "frog", "goat", "goldfish", "goose", "hamster", "hedgehog", "heron", "horse", "kangaroo", "koala",
    "ladybug", "lamb", "lemur", "lion cub", "lizard", "llama", "mole", "monkey", "mouse", "newt",
    "otter", "owl", "panda", "parrot", "penguin", "pony", "porcupine", "puppy", "raccoon", "seal",
    "sheep", "skunk", "snail", "sparrow", "squirrel", "starfish", "swan", "tiger cub", "toad", "turtle",
    "whale", "woodpecker", "zebra", "ant", "bat", "calf", "chick", "cub", "dragonfly",
    "fawn", "flamingo", "gecko", "guinea pig", "hare", "kitten", "moth", "octopus", "peacock", "robin",
    "seahorse", "slug", "yak", "canary", "pony foal",
]

SETTINGS = [
    "apple orchard", "art room", "bakery", "beach path", "book nook", "botanical garden", "bridge", "bus stop",
    "camp cabin", "city library", "classroom", "cloudy hill", "community garden", "courtyard", "creek bank",
    "desert trail", "dining room", "farm lane", "ferry dock", "flower shop", "forest path", "garden shed",
    "greenhouse", "harbor", "ice rink", "kitchen table", "lake shore", "lantern festival", "laundry room",
    "market stall", "meadow", "moonlit porch", "museum hall", "music room", "neighborhood lane", "old tower",
    "picnic blanket", "playroom", "pond edge", "post office", "quiet street", "reading corner", "river bend",
    "school hallway", "seaside pier", "snowy yard", "soccer field", "sunny balcony", "tea shop", "treehouse",
    "vegetable patch", "village square", "wagon path", "window seat", "workshop", "zoo walkway", "train station",
    "rainy sidewalk", "hilltop path", "craft table", "puddle lane", "storybook tent", "map room", "puppet stage",
    "tiny greenhouse", "clay studio", "summer porch", "winter garden", "rocky beach", "pine trail", "cottage gate",
    "chalk courtyard", "kite field", "berry patch", "science corner", "toy shelf", "sleepy attic", "painted fence",
    "water fountain", "birdwatching hide", "seed shop", "blanket fort",
]

OBJECTS = [
    "acorn", "apron", "backpack", "basket", "bell", "blue ribbon", "book", "bottle", "brush", "button",
    "candle", "chalk", "compass", "cup", "drum", "feather", "flag", "flashlight", "flowerpot", "glove",
    "green scarf", "jar", "kite", "ladder", "lantern", "leaf", "letter", "magnifying glass", "map", "marble",
    "mittens", "mug", "notebook", "paintbrush", "paper boat", "pencil", "pebble", "photo", "pillow", "pinwheel",
    "plant pot", "postcard", "puzzle", "quilt", "rain hat", "rope", "seed packet", "shell", "silver spoon", "sketchbook",
    "sock", "spade", "spool", "star sticker", "stone", "storybook", "string", "sun hat", "teacup", "teddy bear",
    "ticket", "tiny key", "toolbox", "toy train", "umbrella", "watering can", "wooden block", "wool hat", "yellow ball",
    "yo-yo", "zip pouch", "clay bowl", "paper crown", "red bucket", "soft blanket", "tin whistle", "wooden flute",
    "cookie tin", "garden fork", "blue button", "ruler", "paper lantern", "cloth bag",
]

LESSONS = [
    "ask before borrowing", "be patient while learning", "clean up after a mistake", "comfort a worried friend",
    "finish a promise", "listen before answering", "notice small kindness", "practice gently", "share credit",
    "speak honestly", "take turns", "try again after slipping", "use quiet courage", "welcome someone new",
    "work together", "apologize clearly", "ask for help", "be careful with fragile things", "care for nature",
    "choose kind words", "help without showing off", "include a shy friend", "keep trying", "learn from a mix-up",
    "look for a fair solution", "make room for others", "notice feelings", "offer thanks", "prepare ahead",
    "respect a different idea", "return what was found", "solve one small problem", "tell the truth kindly",
    "use imagination wisely", "wait for a turn", "watch where you step", "welcome a change", "work slowly and well",
    "protect a small creature", "repair something broken", "be brave in a new place", "listen to instructions",
    "make a simple plan", "calm down before speaking", "help a younger child", "notice when someone is tired",
    "keep a shared space neat", "make a fair trade", "try a new food politely", "save a surprise for later",
    "say sorry and mean it", "respect a quiet moment",
]

EXPLANATION_TOPICS = [
    "why rain falls", "how seeds grow", "why shadows move", "how magnets pull", "why soap makes bubbles",
    "how birds build nests", "why leaves change color", "how bread rises", "why we brush teeth", "how bicycles balance",
    "why the moon looks different", "how clouds form", "why snow melts", "how a thermometer works", "why boats float",
    "how bees help flowers", "why we sleep", "how sound travels", "why ice is slippery", "how plants drink water",
    "why stars twinkle", "how a kite stays up", "why wheels roll", "how a zipper closes", "why cats purr",
    "how a clock tells time", "why oceans have waves", "how a rainbow appears", "why metal feels cold", "how glue sticks",
    "why exercise helps bodies", "how a bridge holds weight", "why apples turn brown", "how glasses help eyes",
    "why fire needs air", "how a camera takes pictures", "why some animals migrate", "how compost helps soil",
    "why hands wrinkle in water", "how a pencil writes", "why echo happens", "how sunscreen protects skin",
    "why balloons float", "how wheels on skates work", "why we recycle", "how a lock opens", "why popcorn pops",
    "how a musical note is made", "why fish have gills", "how a parachute slows down", "why roots spread out",
    "how a simple pulley helps",
]

QA_TOPICS = [
    "packing a lunch", "finding a lost mitten", "watering a plant", "choosing a book", "crossing a street safely",
    "helping with dishes", "sharing crayons", "feeding a pet", "cleaning muddy shoes", "getting ready for bed",
    "visiting a library", "planting beans", "making a card", "joining a game", "listening to a teacher",
    "waiting in line", "using an umbrella", "taking care of a toy", "asking for a turn", "saying thank you",
    "sorting recycling", "helping a neighbor", "washing hands", "trying a puzzle", "saving a snack",
    "wearing a helmet", "choosing warm clothes", "tidying a shelf", "finding a seat", "telling the truth",
    "using indoor voices", "helping a younger child", "watching a parade", "taking turns on a swing", "making soup",
    "following a map", "drawing a picture", "learning a song", "borrowing a pencil", "looking after a seedling",
    "calming a friend", "reading a sign", "packing a backpack", "keeping a promise", "checking the weather",
    "feeding birds", "setting a table", "cleaning a spill", "opening a jar", "walking a puppy", "using a bookmark",
    "choosing a safe path",
]

DIALOGUE_SITUATIONS = [
    "two friends planning a picnic", "a child asking a librarian for help", "a fox and owl fixing a kite",
    "siblings choosing a game", "a teacher explaining a garden task", "a baker and child counting rolls",
    "a turtle asking a heron for directions", "friends deciding how to share paint", "a parent helping with shoelaces",
    "neighbors returning a lost key", "a child welcoming a new classmate", "a puppy learning to wait",
    "a bird asking about a storm", "friends apologizing after a bump", "a shopkeeper finding a missing button",
    "a child asking why leaves fall", "two children making a fair rule", "a grandmother teaching a song",
    "a coach praising careful practice", "a cat and mouse building a bridge", "a child inviting a shy friend",
    "a family packing for rain", "a farmer explaining seeds", "two friends solving a puzzle", "a bus driver giving directions",
    "a child offering help with a bag", "a squirrel asking to borrow a basket", "a child and robot sorting toys",
    "friends choosing a bedtime story", "a child thanking a nurse", "two animals cleaning a puddle",
    "a librarian suggesting a quiet voice", "a child asking for more time", "a friend sharing a snack",
    "two children repairing a model", "a parent explaining patience", "a penguin asking about warm clothes",
    "a child returning a scarf", "friends preparing a small play", "a child checking on a sad friend",
    "two animals making a map", "a child asking about a rainbow", "friends taking turns with a drum",
    "a child talking to a gardener", "a class planning a clean-up", "a child explaining a mistake",
    "two friends building a blanket fort", "a child asking to join a drawing", "a helper giving calm instructions",
    "a child and uncle setting a table", "two animals sharing a lantern",
]

REASONING_SITUATIONS = [
    "choosing which wet item should dry first", "figuring out who gets the next turn", "sorting snacks by color",
    "deciding which path is shorter", "finding what is missing from a picnic basket", "matching animals to homes",
    "choosing the safest tool for a task", "ordering morning steps", "counting seats for friends",
    "deciding how to split crayons fairly", "noticing why a plant drooped", "picking clothes for rainy weather",
    "finding the quietest place to read", "choosing a container that will not spill", "deciding which toy can float",
    "working out who arrived first", "planning how to carry two bags", "choosing a light for a dark room",
    "finding the best time to water flowers", "sorting recycling into bins", "choosing a fair team order",
    "deciding how many cups are needed", "matching keys to labels", "finding why footprints stopped",
    "choosing the strongest bridge material", "ordering story events", "deciding which shelf fits a book",
    "figuring out why a kite fell", "choosing a calm way to solve a disagreement", "finding the warmer scarf",
    "deciding what to do before crossing", "matching sounds to instruments", "choosing who needs help first",
    "planning steps to clean a spill", "finding which seed grew tallest", "deciding how to share three apples",
    "choosing the next puzzle piece", "ordering jars from smallest to largest", "finding why a door will not close",
    "choosing a route around a puddle", "deciding which note belongs in a thank-you card", "matching shadows to objects",
    "choosing a snack for a picnic rule", "finding the missing number in a simple pattern", "deciding when to ask for help",
    "choosing a safe place for a candle", "sorting buttons by size", "finding the cause of a muddy trail",
    "choosing the best way to carry eggs", "ordering steps to mail a letter", "deciding which friend should speak next",
]

STRUCTURED_TASKS = [
    "three-item checklist", "two-column table", "numbered plan", "simple JSON object", "yes-no decision with reason",
    "morning routine list", "safety checklist", "lost-and-found note fields", "kindness plan", "story map",
    "cause-effect pairs", "pros and cons list", "materials list", "packing list", "recipe steps", "reading log",
    "weather plan", "garden care chart", "pet care schedule", "emotion-to-action map", "turn-taking rules",
    "cleanup checklist", "question-answer pairs", "character facts", "event timeline", "problem-solution table",
    "sorting categories", "goal tracker", "thank-you note outline", "classroom rule card", "tiny poem template",
    "observation report", "draw-and-label prompt",
]

CREATIVE_TYPES = [
    "riddle", "short poem", "lullaby verse", "pretend postcard", "friendly sign", "mini song", "title list",
    "make-believe recipe", "treasure clue", "birthday wish", "thank-you note", "garden chant", "weather rhyme",
    "animal motto", "tiny play idea", "magic object description", "gentle joke", "classroom poster line",
    "bedtime whisper story", "storybook title", "kindness pledge", "festival invitation", "toy description",
    "map clue", "secret club rule", "drawing prompt", "paper crown message", "puppet introduction", "goodnight note",
    "simple chant", "friendly warning sign", "lost item notice",
]

SUMMARIZATION_TEMPLATES = [
    "A child finds a {object} in the {setting}, asks who lost it, and returns it before lunch.",
    "Two friends visit the {setting}, disagree about a {object}, then choose a fair turn-taking plan.",
    "A {animal} notices a small problem near the {setting} and gets help from {name}.",
    "During a rainy morning, {name} carries a {object}, helps a neighbor, and learns to slow down.",
    "At the {setting}, a group cleans up, sorts useful items, and thanks the quiet helper.",
    "{name} brings a {object} to the {setting}, but a small mix-up teaches everyone to label their things.",
    "A {animal} waits near the {setting} while children make a plan to fix a wobbly sign.",
    "The class visits the {setting}, observes a {object}, and writes down three careful facts.",
    "{name} and a {animal} lose a {object}, retrace their steps, and find it beside a bench.",
    "A storm changes the plan at the {setting}, so the children move inside and help set up a quiet game.",
    "A younger child feels nervous at the {setting}, and {name} uses a {object} to make a friendly welcome.",
    "The group wants to decorate the {setting}, but they first agree on a safe place for each {object}.",
    "A {animal} makes a mess with a {object}, then helps clean it before anyone gets upset.",
    "{name} hears a strange sound near the {setting}, checks carefully, and finds a harmless explanation.",
    "Friends bring snacks to the {setting}, count what they have, and share so everyone gets some.",
    "A missing {object} delays the activity, but a {animal} remembers where it was last seen.",
    "{name} practices a new skill at the {setting}, makes one mistake, and tries again calmly.",
    "The children find footprints near the {setting}, compare clues, and learn which animal passed by.",
    "A {object} breaks during play, so {name} and a {animal} repair it with patience.",
    "Everyone wants the same seat at the {setting}, and the children solve it with a turn list.",
    "{name} notices trash near the {setting}, gathers friends, and leaves the place cleaner than before.",
    "A {animal} cannot reach a {object}, so the children choose a safe helper and a steady step.",
    "The lights go out at the {setting}, and {name} uses calm words while an adult finds a lantern.",
    "A new child arrives at the {setting}, and the group explains the game in simple steps.",
    "{name} finds a note tucked under a {object}, follows its clue, and thanks the person who wrote it.",
    "The children prepare a show at the {setting}, forget one {object}, and make a simple replacement.",
    "A {animal} is afraid of a loud noise, so {name} waits nearby and speaks softly.",
    "Friends sort a basket of objects at the {setting}, putting each useful thing where it belongs.",
    "{name} wants to hurry through the {setting}, but a careful friend points out a slippery spot.",
    "A small plant at the {setting} droops, and the children figure out what care it needs.",
    "The group builds a model with a {object}, tests it, and changes one part to make it stronger.",
    "A {animal} follows a trail of crumbs near the {setting}, and {name} cleans them up safely.",
    "The children make a map of the {setting}, mark the important places, and use it to help a visitor.",
    "{name} borrows a {object}, forgets to return it, and fixes the mistake with an apology.",
    "A picnic at the {setting} is almost spoiled by wind, but the friends weigh down the blanket.",
    "The children hear that someone is sad, bring a small {object}, and invite the person to join them.",
    "A {animal} watches children paint at the {setting}, and they set rules to keep the paint tidy.",
    "{name} discovers that two friends need the same {object}, so they plan a fair schedule.",
    "The group studies clouds from the {setting}, draws what they see, and predicts rain.",
    "A child drops a {object} into a puddle, and friends help dry it without blaming anyone.",
    "{name} and a {animal} follow a safe path through the {setting} and avoid stepping on flowers.",
    "The children prepare thank-you cards at the {setting}, each naming one helpful action.",
    "A noisy game bothers readers near the {setting}, and the children choose a better place to play.",
    "{name} finds a tiny creature near a {object}, watches respectfully, and leaves it alone.",
    "The class uses a {object} for a science activity, records results, and cleans the table.",
    "A friend feels left out at the {setting}, so {name} changes the rules to include everyone.",
    "The children make soup at the {setting}, count ingredients, and wait until it is cool.",
    "A {animal} blocks the path near the {setting}, and the group waits instead of rushing.",
    "{name} learns that a {object} belongs to the whole class and should be handled gently.",
    "The group plans a surprise at the {setting}, keeps it kind, and thanks the helper afterward.",
    "A child misunderstands a rule about a {object}, asks a question, and learns the right way.",
    "{name} sees a friend working alone near the {setting} and offers one helpful action.",
    "The children compare two routes to the {setting}, choose the safer one, and arrive together.",
    "A {animal} hides behind a {object}, and the children give it space until it feels safe.",
    "The group gathers leaves at the {setting}, sorts them by shape, and leaves living plants untouched.",
    "{name} spills water near a {object}, warns others, and helps wipe the floor dry.",
]

INSTRUCTION_VERBS = {
    "story_writing": ["Write", "Create", "Tell", "Draft", "Make"],
    "story_continuation": ["Continue", "Finish", "Extend", "Complete", "Add an ending to"],
    "explanation_for_children": ["Explain", "Describe", "Teach", "Show", "Help a child understand"],
    "simple_qa": ["Answer", "Respond to", "Give a simple answer to", "Help with", "Reply to"],
    "dialogue": ["Write a dialogue for", "Create a conversation about", "Draft lines for", "Make a short chat about"],
    "summarization": ["Summarize", "Retell briefly", "Give a short summary of", "Explain the main idea of"],
    "simple_reasoning": ["Reason through", "Solve", "Figure out", "Explain the best choice for"],
    "structured_output": ["Fill out", "Create", "Make", "Format"],
    "creative_generation": ["Invent", "Create", "Write", "Make"],
}


def category_counts(count: int) -> dict[str, int]:
    if count <= 0:
        raise ValueError("count must be positive.")
    raw = {key: count * value / 1000 for key, value in TARGET_DISTRIBUTION_1000.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remainder = count - sum(counts.values())
    ordered = sorted(raw, key=lambda key: (raw[key] - counts[key], TARGET_DISTRIBUTION_1000[key]), reverse=True)
    for key in ordered[:remainder]:
        counts[key] += 1
    return counts


def _pick(items: list[str], index: int, rng: random.Random) -> str:
    return items[(index + rng.randrange(len(items))) % len(items)]


def _common_constraints(min_words: int, max_words: int) -> dict[str, Any]:
    return {
        "min_words": min_words,
        "max_words": max_words,
        "avoid_openings": BANNED_OPENINGS,
        "avoid_names": sorted(BANNED_NAMES),
        "must_include": ["one small problem", "one helpful action", "a clear ending"],
    }


def _paragraph_template(index: int, name: str, animal: str, setting: str, obj: str) -> str:
    template = SUMMARIZATION_TEMPLATES[index % len(SUMMARIZATION_TEMPLATES)]
    return template.format(name=name, animal=animal, setting=setting, object=obj)


def make_seed_row(seed_number: int, task_type: str, rng: random.Random) -> dict[str, Any]:
    seed_id = f"seed_{seed_number:06d}"
    name = _pick(NAMES, seed_number, rng)
    animal = _pick(ANIMALS, seed_number * 3, rng)
    setting = _pick(SETTINGS, seed_number * 5, rng)
    obj = _pick(OBJECTS, seed_number * 7, rng)
    lesson = _pick(LESSONS, seed_number * 11, rng)
    verb = _pick(INSTRUCTION_VERBS[task_type], seed_number, rng)
    row: dict[str, Any] = {
        "seed_id": seed_id,
        "target_sft_id": f"xm_sft_{seed_number:06d}",
        "task_type": task_type,
        "input_blueprint": "",
        "tone": "simple, warm, child-friendly",
    }
    if task_type == "story_writing":
        row.update(
            {
                "instruction_blueprint": f"{verb} a child-friendly story seed {seed_id} about {name} and a {animal} in a {setting}; the lesson is to {lesson}.",
                "topic": f"{name} helps a {animal}",
                "setting": setting,
                "characters": [name, animal],
                "lesson": lesson,
                "constraints": _common_constraints(30, 140),
            }
        )
    elif task_type == "story_continuation":
        opening = f"{name} found a {obj} beside the {setting}. A {animal} looked worried and pointed toward a new clue."
        row.update(
            {
                "instruction_blueprint": f"{verb} this story seed {seed_id} with a clear ending and no copied opening.",
                "input_blueprint": opening,
                "topic": f"continue a {animal} story",
                "setting": setting,
                "characters": [name, animal],
                "lesson": lesson,
                "constraints": _common_constraints(35, 130),
            }
        )
    elif task_type == "explanation_for_children":
        topic = _pick(EXPLANATION_TOPICS, seed_number * 13, rng)
        row.update(
            {
                "instruction_blueprint": f"{verb} {topic} for a young child in seed {seed_id}, using plain words and one everyday example.",
                "topic": topic,
                "constraints": {"min_words": 30, "max_words": 110, "avoid_names": sorted(BANNED_NAMES), "must_include": ["simple cause", "concrete example"]},
            }
        )
    elif task_type == "simple_qa":
        topic = _pick(QA_TOPICS, seed_number * 17, rng)
        row.update(
            {
                "instruction_blueprint": f"{verb} a child's question about {topic} in seed {seed_id}.",
                "topic": topic,
                "constraints": {"min_words": 20, "max_words": 80, "must_include": ["direct answer", "kind reason"]},
            }
        )
    elif task_type == "dialogue":
        situation = _pick(DIALOGUE_SITUATIONS, seed_number * 19, rng)
        row.update(
            {
                "instruction_blueprint": f"{verb} {situation} for seed {seed_id}; use 4 to 8 short turns.",
                "topic": situation,
                "characters": [name, animal],
                "constraints": {"min_words": 35, "max_words": 130, "must_include": ["speaker names", "friendly resolution"]},
            }
        )
    elif task_type == "summarization":
        paragraph = _paragraph_template(seed_number, name, animal, setting, obj)
        row.update(
            {
                "instruction_blueprint": f"{verb} the paragraph for seed {seed_id} in one or two simple sentences.",
                "input_blueprint": paragraph,
                "topic": "short child-safe paragraph summary",
                "constraints": {"min_words": 15, "max_words": 55, "must_include": ["main problem", "ending"]},
            }
        )
    elif task_type == "simple_reasoning":
        situation = _pick(REASONING_SITUATIONS, seed_number * 23, rng)
        row.update(
            {
                "instruction_blueprint": f"{verb} {situation} for seed {seed_id}; give the answer and one short reason.",
                "topic": situation,
                "constraints": {"min_words": 20, "max_words": 80, "must_include": ["answer", "reason"]},
            }
        )
    elif task_type == "structured_output":
        structured_type = _pick(STRUCTURED_TASKS, seed_number * 29, rng)
        row.update(
            {
                "instruction_blueprint": f"{verb} a {structured_type} for seed {seed_id} about {setting} and a {obj}.",
                "topic": structured_type,
                "setting": setting,
                "constraints": {"min_words": 20, "max_words": 100, "format": structured_type, "must_include": ["clear labels", "short entries"]},
            }
        )
    elif task_type == "creative_generation":
        creative_type = _pick(CREATIVE_TYPES, seed_number * 31, rng)
        row.update(
            {
                "instruction_blueprint": f"{verb} a {creative_type} for seed {seed_id} involving a {animal}, a {obj}, and a {setting}.",
                "topic": creative_type,
                "setting": setting,
                "characters": [animal],
                "constraints": {"min_words": 10, "max_words": 80, "avoid_names": sorted(BANNED_NAMES), "must_include": ["playful but clear language"]},
            }
        )
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
    return row


def make_seed_rows(*, count: int, start_id: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    counts = category_counts(count)
    task_types = [task_type for task_type, task_count in counts.items() for _ in range(task_count)]
    rng.shuffle(task_types)
    rows = [make_seed_row(start_id + offset, task_type, rng) for offset, task_type in enumerate(task_types)]
    instruction_counts = Counter(row["instruction_blueprint"] for row in rows)
    duplicate_instructions = [text for text, text_count in instruction_counts.items() if text_count > 1]
    if duplicate_instructions:
        raise RuntimeError(f"Generated duplicate instruction_blueprint values: {duplicate_instructions[:3]}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_seed_output(*, rows: list[dict[str, Any]], out: Path | None, out_dir: Path | None, shard_size: int | None) -> list[Path]:
    if out_dir is not None:
        if shard_size is None or shard_size <= 0:
            raise ValueError("--shard-size must be positive when --out-dir is used.")
        paths = []
        for shard_index, start in enumerate(range(0, len(rows), shard_size), start=1):
            shard_rows = rows[start : start + shard_size]
            path = out_dir / f"sft_seeds_{shard_index:04d}.jsonl"
            _write_jsonl(path, shard_rows)
            paths.append(path)
        return paths
    if out is None:
        raise ValueError("Either --out or --out-dir is required.")
    _write_jsonl(out, rows)
    return [out]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic seed prompts for external Xiaomi/OpenCode SFT generation.")
    parser.add_argument("--out", default=None, help="Single output JSONL path.")
    parser.add_argument("--out-dir", default=None, help="Directory for JSONL shards.")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--profile", default="v0_4_child_simple", choices=["v0_4_child_simple"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shard-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = make_seed_rows(count=args.count, start_id=args.start_id, seed=args.seed)
    unknown = set(row["task_type"] for row in rows) - ALLOWED_SFT_TASK_TYPES
    if unknown:
        raise RuntimeError(f"Generated unsupported task types: {sorted(unknown)}")
    paths = write_seed_output(
        rows=rows,
        out=Path(args.out) if args.out else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        shard_size=args.shard_size,
    )
    print(json.dumps({"rows": len(rows), "category_counts": dict(Counter(row["task_type"] for row in rows)), "paths": [str(path) for path in paths]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
