"""
Basic models and layers used for trajectory prediction.

## Public Classes

- `Model`:
    The basic model class that contains log-related and manager-member-related
    methods. Subclass it to customize trajectory prediction models.

## Modules

- `layers`: Useful model layers.
- `process`: Pre/post-process layers and models for trajectories.
- `transformer`: The basic transformer backbone for trajectory prediction.
"""

from . import layers, process, transformer
from .__model import Model
