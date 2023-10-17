"""
Argument-related classes and constants.

## Public Classes

- `Args`:
    The basic class to manage all arguments for training or testing
    trajectory prediction models. It contains several builtin args
    for managing the training and testing process for training
    trajectory prediction models.

- `EmptyArgs`:
    The basic class to manage any args. It contains no training or
    test args, and it should be subclassed when using.

## Constants

- `DYNAMIC`:
    A flag to indicate some argument's type.
    Args with argtype=`dynamic` can be changed anytime.
    The program will try to first parse inputs from the terminal and 
    then try to load from the saved JSON file.
    
- `STATIC`:
    A flag to indicate some argument's type.
    Args with argtype=`static` can not be changed once after training.
    When testing the model, the program will not parse these args to
    overwrite the saved values.

- `TEMPORARY`:
    A flag to indicate some argument's type.
    Args with argtype=`temporary` will not be saved into JSON files.
    The program will parse these args from the terminal at each time.

## Variables

- `ARG_ALIAS`:
    A dictionary that saves all aliases for running args.

## Public Methods:

- `add_arg_alias`:
    Add new arg alias.

- `parse_arg_alias`:
    Parse arg alias from a list of args.
"""

from ..__root.__argsManager import ArgsManager as EmptyArgs
from ..__root.__argsManager import register_new_args
from .__args import (ARG_ALIAS, DYNAMIC, STATIC, TEMPORARY, Args,
                     add_arg_alias, parse_arg_alias)
