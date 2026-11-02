"""Turn a written prompt into a scene and its settings.

The prompt picks and tunes a scene from the catalogue rather than generating code:
a fixed set of scenes that are known to loop and known to read as a background is
more useful than free-form generation that may do neither.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .scenes import SCENES, Scene

PALETTE_WORDS: Dict[str, Tuple[str, ...]] = {
    "ice": ("blue", "ice", "icy", "cold", "winter", "frozen", "arctic", "cyan", "glacier"),
    "amber": ("amber", "orange", "gold", "golden", "warm", "sunset", "desert", "sand", "dusk"),
    "green": ("green", "emerald", "terminal", "jungle", "forest", "lime", "toxic"),
    "ember": ("red", "fire", "ember", "lava", "volcano", "hot", "crimson", "blood", "mars"),
    "violet": ("purple", "violet", "magenta", "pink", "neon", "synthwave", "vapor", "ultraviolet"),
    "mono": ("grey", "gray", "white", "mono", "monochrome", "black", "silver", "ink"),
}

FASTER = ("fast", "quick", "rapid", "racing", "speeding", "frantic", "urgent")
SLOWER = ("slow", "gentle", "calm", "lazy", "drifting", "soft", "quiet", "still")
DENSER = ("heavy", "dense", "thick", "busy", "swarm", "storm", "pouring", "crowded")
SPARSER = ("sparse", "light", "few", "minimal", "faint", "empty", "bare", "distant")


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def scene_from_prompt(text: str) -> Tuple[Scene, str]:
    """Pick a scene from the prompt and tune it. Returns the scene and what it matched."""
    words = _words(text)
    bag = set(words)

    scored = []
    for name, cls in SCENES.items():
        hits = bag & set(cls.tags)
        scored.append((len(hits), name, sorted(hits)))
    scored.sort(key=lambda row: (-row[0], row[1]))
    best, name, hits = scored[0]

    reasons = []
    if best:
        reasons.append(f"matched {', '.join(hits)}")
    else:
        name = "nebula"
        reasons.append("nothing matched, using nebula")

    scene = SCENES[name]()

    for palette, keys in PALETTE_WORDS.items():
        found = bag & set(keys)
        if found and hasattr(scene, "palette"):
            scene.palette = palette
            reasons.append(f"{palette} from {sorted(found)[0]}")
            break

    if hasattr(scene, "speed"):
        if bag & set(FASTER):
            scene.speed *= 2.0
            reasons.append("faster")
        elif bag & set(SLOWER):
            scene.speed *= 0.5
            reasons.append("slower")

    if hasattr(scene, "density"):
        if bag & set(DENSER):
            scene.density *= 1.7
            reasons.append("denser")
        elif bag & set(SPARSER):
            scene.density *= 0.5
            reasons.append("sparser")

    return scene, f"{name}: " + "; ".join(reasons)
