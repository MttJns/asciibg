"""Character ramps, sparse to dense."""

from __future__ import annotations

RAMPS = {
    # ascii.rest's shade ramp, with a leading space so a background can go fully dark.
    "shade": " .:-=+*#%@",
    "dot": " ·•●",
    "block": " ░▒▓█",
    "soft": " .,:;ox%#@",
}

DEFAULT_RAMP = "shade"


def ramp(name: str) -> str:
    """Return the ramp named `name`, sparse character first."""
    try:
        return RAMPS[name]
    except KeyError:
        raise KeyError(f"unknown ramp {name!r}; choose from {', '.join(sorted(RAMPS))}") from None
