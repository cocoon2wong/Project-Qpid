"""
Argument-related classes and constants.

## Public Classes

- `Args`:
    The basic class to manage all arguments for training or testing
    trajectory prediction models. It contains several builtin args
    for managing the training and testing process for training
    trajectory prediction models.

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
"""

from .__args import DYNAMIC, STATIC, TEMPORARY, Args
