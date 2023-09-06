"""
Root classes for the Qpid package.

## Public Classes

- `BaseObject`:
    The basic father class for all Qpid classes.
    It contains several log-related methods.
- `ArgsManager`:
    The basic class to manage all arguments used in all Qpid classes.
"""

from .__argsManager import ArgsManager
from .__baseObject import BaseObject
