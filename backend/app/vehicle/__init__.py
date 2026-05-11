"""Vehicle-monitoring helpers ported from the surveillance-system fork.

This package is a thin domain layer parallel to the existing pedestrian code
in store.py / inference.py. It exposes:

- los: V/C ratio thresholds + PCE multipliers + LOS letter grades (HCM-style)
- gates: 5 fixed campus gates with both geographic (lat/lng) and pixel
  coordinates plus flow-group ("In" entrance vs "Out" exit) classification
- counting: utilities for reading the JSON line-crossing configs that ship
  with Occlusion-Robust-RTDETR
"""

from . import counting, gates, los

__all__ = ["counting", "gates", "los"]
