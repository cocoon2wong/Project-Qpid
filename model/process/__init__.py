"""
Pre/post-process-related classes and constants.

## Public Classes

- `BaseProcessLayer`:
    The basic process layer. Subclass it to make customized process layers.

## Constants

- `Move`, `Rotate`, `Scale`:
    Simplest preprocess and postprocess methods used on trajectories.

- `PROCESS_TYPES`:
    Type identifiers for `[Move, Rotate, Scale]`.

- `PROCESS_DICT`:
    Get one of `[Move, Rotate, Scale]` by their names recorded in `PROCESS_TYPES`.
"""

from ...constant import PROCESS_TYPES
from .__base import BaseProcessLayer
from .__move import Move
from .__processModel import PROCESS_DICT, ProcessModel
from .__rotate import Rotate
from .__scale import Scale
