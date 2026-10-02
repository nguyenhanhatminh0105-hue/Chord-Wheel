"""A hand-built right hand, upright, with chosen fingers raised (image coordinates, y down)."""
import numpy as np

MCP = {"index": (-0.30, -1.00), "middle": (0.00, -1.00), "ring": (0.25, -0.95), "little": (0.45, -0.85)}
BASE = {"index": 5, "middle": 9, "ring": 13, "little": 17}


def synthetic_hand(raised=(), center=(0.5, 0.7), size=0.1) -> np.ndarray:
    p = np.zeros((21, 2))
    p[1], p[2], p[3] = (-0.30, -0.20), (-0.55, -0.40), (-0.70, -0.60)      # thumb CMC, MCP, IP
    p[4] = (-1.00, -0.85) if "thumb" in raised else (-0.25, -0.75)        # thumb tip
    for name, base in BASE.items():
        mx, my = MCP[name]
        p[base] = (mx, my)                       # knuckle
        p[base + 1] = (mx, my - 0.35)            # middle joint (PIP)
        if name in raised:
            p[base + 2], p[base + 3] = (mx, my - 0.60), (mx, my - 0.80)   # straight up
        else:
            p[base + 2], p[base + 3] = (mx, my - 0.15), (mx, my + 0.15)   # curled toward the palm
    return p * size + np.asarray(center)
