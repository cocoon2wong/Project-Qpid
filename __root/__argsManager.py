"""
@Author: Conghao Wong
@Date: 2022-11-11 12:41:16
@LastEditors: Conghao Wong
@LastEditTime: 2025-04-22 10:30:09
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import json
import os
from typing import Any, TypeVar

from ..constant import ARG_TYPES
from ..utils import ARGS_FILE_NAME, dir_check
from .__baseObject import BaseObject

DYNAMIC = ARG_TYPES.DYNAMIC
STATIC = ARG_TYPES.STATIC
TEMPORARY = ARG_TYPES.TEMPORARY

T = TypeVar('T')


class ArgsManager(BaseObject):
    """
    Args Manager
    ---
    The basic class to manage all args when training or testing models.
    """

    _ignore_value_check = False

    def __init__(self, terminal_args: list[str] | None = None,
                 is_temporary=False) -> None:
        """
        :param terminal_args: A set of args that received from the user input.
        :param is_temporary: Controls whether this `Args` object is a set of\
            temporary args or a set of args used for training. Temporary args\
            will not initialize the `log_dir`.
        """
        super().__init__(name='Args Manager')

        # Init variables
        self._terminal_args = terminal_args
        self._is_temporary = is_temporary
        self._init_done = False
        self._args_need_initialize: list[str] = []

        # Init subargs
        self._subargs_types: dict[str, Any] = {}
        self._subargs_dict: dict[str, Any] = {}

        # Args that load from the saved JSON file
        self._args_load: dict[str, Any] = {}

        # Args obtained from terminal
        self._args_runnning: dict[str, Any] = {}

        # Args that are set manually
        self._args_manually: dict[str, Any] = {}

        # The default args (manually)
        self._args_default_manually: dict[str, Any] = {}

        # The default args (default)
        self._args_default: dict[str, Any] = {}

        # Args' descriptions that appear in the model summary
        self._args_desc: dict[str, str | tuple[str, str]] = {}

        # A list to save all registered args' names
        self._arg_list = []

        # Types of all args, e.g. `STATIC`.
        self._arg_type: dict[str, str] = {}

        # Short names of all args.
        # Keys: short names.
        # Values: Full names.
        self._arg_short_name: dict[str, str] = {}

        # Aliases of all args
        self._arg_aliases: dict[str, list[str]] = {}

        # Register all args used in this object
        self._visit_args()
        self._init_done = True

        # Load terminal args
        if terminal_args:
            self.load_args_from_terminal(terminal_args)

        # Load json args
        if (l := self.load) != 'null':
            from .. import sys_mgr
            if l in sys_mgr.static_models:
                self._set_default('model', l)
            else:
                self.load_args_from_json(l)

        # Restore reference args before training and testing
        if self.restore_args != 'null':
            self.load_args_from_json(self.restore_args, 'default')

        # Run initialize methods
        self._init_all_args()
        self._visit_args()

    def log(self, s: str, level: str = 'info',
            raiseError: type[BaseException] | None = None):
        if self._ignore_value_check and raiseError:
            self.log(f'`{raiseError}` ignored.')
            raiseError = None
        return super().log(s, level, raiseError)

    @property
    def verbose(self) -> int:
        """
        Controls whether to print verbose logs and outputs to the terminal.
        """
        return self._arg('verbose', 0, argtype=TEMPORARY, short_name='v')

    @property
    def load(self) -> str:
        """
        Folder to load model weights (to test). If it is set to `null`, the
        training manager will start training new models according to other
        reveived args. NOTE: Leave this arg to `null` when training new models.
        """
        return self._arg('load', 'null', argtype=TEMPORARY, short_name='l')

    @property
    def restore_args(self) -> str:
        """
        Path to restore the reference args before training.
        It will not restore any args if `args.restore_args == 'null'`.
        """
        return self._arg('restore_args', 'null', argtype=TEMPORARY)

    @property
    def _verbose_mode(self) -> bool:
        if self.verbose and not self._is_temporary:
            return True
        else:
            return False

    def _init_all_args(self):
        if self._verbose_mode:
            self.log('Basic funtions initialized.')

    def register_subargs(self, arg_type: type[T], name: str) -> T:
        """
        Register new args that are used in extra mods to the current args.
        """
        if ((not name in self._subargs_types.keys()) or
                (not name in self._subargs_dict.keys())):
            if not issubclass(arg_type, ArgsManager):
                raise TypeError(arg_type)

            subargs = arg_type(terminal_args=self._terminal_args,
                               is_temporary=True)
            self._subargs_types[name] = arg_type
            self._subargs_dict[name] = subargs

        return self._subargs_dict[name]

    @classmethod
    def get_args_names(cls) -> list[str]:
        """
        Get a list of names of all args used in this class.
        """
        a = cls
        items = []
        while a != BaseObject:
            items.append(a.__dict__.items())
            a = a.__bases__[0]

        return list(set([i for item in items for i, v in item
                         if ((isinstance(v, property)) and
                             (not i.startswith('_')))]))

    def _visit_args(self):
        """
        Visit all args.
        """
        for arg_name in self.get_args_names():
            getattr(self, arg_name)

    def load_args_from_json(self, dir_path: str, target='load'):
        """
        Load args from the saved JSON file.

        :param dir_path: Path to the folder of the JSON file.
        :param target: Target dictionary to load, can be `'load'` or `'default'`.
        """
        try:
            with open(os.path.join(dir_path, ARGS_FILE_NAME), 'r') as f:
                json_dict = json.load(f)

            if target == 'load':
                self._args_load = json_dict
            elif target == 'default':
                self._args_default_manually = json_dict
            else:
                raise ValueError(target)

        except:
            self.log(f'Failed to load args from `{dir_path}`.',
                     level='error', raiseError=ValueError)

        return self

    def load_args_from_terminal(self, argv: list[str]):
        """
        Load args from user's inputs.
        """
        dic = {}

        index = 0
        name = None
        while True:
            try:
                if argv[index].startswith('--'):
                    name = argv[index][2:]
                    name = name.replace('-', '_')

                elif argv[index].startswith('-'):
                    name = argv[index][1:]
                    name = name.replace('-', '_')
                    name = self._arg_short_name[name]  # <- may raise KeyError

                else:
                    index += 1
                    continue

                if (index+1 == len(argv) or
                    ((value := argv[index+1]).startswith('-')
                        and value[1] not in '0123456789')):
                    dic[name] = True
                    index += 1
                else:
                    dic[name] = value
                    index += 2

            except IndexError:
                break

            except KeyError:
                index += 1

        self._args_runnning = dic
        return self

    def save_args_as_json(self, target_dir: str):
        """
        Save current args into a JSON file.
        """
        dir_check(target_dir)
        json_path = os.path.join(target_dir, ARGS_FILE_NAME)

        all_args = [self] + list(self._subargs_dict.values())

        names = []
        values = []
        for index, arg in enumerate(all_args):
            names.append(f'#---------- {type(arg).__name__} ----------')
            values.append(index)

            _names = [n for (n, v) in arg._arg_type.items() if v != TEMPORARY]
            _names.sort()

            names += _names
            values += [getattr(arg, n) for n in _names]

        with open(json_path, 'w+') as f:
            json.dump(dict(zip(names, values)), f,
                      separators=(',\n', ':'))

    def _get_args_by_index_and_name(self, index: int, names: list[str]):
        if index == 0:
            dic = self._args_load
        elif index == 1:
            dic = self._args_runnning
        elif index == 99:
            dic = self._args_manually
        elif index == -1:
            dic = self._args_default_manually
        else:
            raise ValueError('Args index not exist.')

        for name in names:
            if name in dic.keys():
                return dic[name]
        return None

    def _set(self, name: str, value: Any, verbose=False):
        """
        Set argument manually.
        """
        self._args_manually[name] = value

        if verbose:
            self.log(f'Arg `{name}` has been set to `{value}`.')

    def _set_default(self, name: str, value: Any, overwrite=True):
        """
        Set default argument values.
        """
        write = True
        if name in self._args_default_manually.keys():
            if not overwrite:
                write = False

        if write:
            self._args_default_manually[name] = value

    def _arg(self, name: str,
             default: T,
             argtype: str,
             short_name: str | None = None,
             need_initialize: bool = False,
             other_names: list[str] | None = None,
             desc_in_model_summary: str | tuple[str, str] | None = None) -> T:
        """
        Get arg from all arg dictionaries according to the priority.

        :param name: Name of the arg, should be the same as the property's name.
        :param default: Default value of the arg.
        :param argtype: Arg type, canbe `STATIC`, `DYNAMIC`, or `TEMPORARY`.
        :param short_name: Short name of the arg. It does not contain a `-`.
        :param need_initialize: Set whether this arg need to be initialized before using.
        :param other_names: A list of other names of this arg. It could be used when \
            loading args from the saved json files.
        :param desc_in_model_summary: Set whether this arg will appear in the model summary. \
            It will not appear when it is set to `None`, and other string values will be \
            used as the description of this arg. If this arg belongs to some arg-group, \
            please pass them in the tuple-like way, like `('ARG GROUP', 'Arg desc')`.
        """
        # Register args before using
        if not self._init_done:
            self._register(name, default, argtype,
                           short_name, other_names,
                           desc_in_model_summary)
            if need_initialize:
                self._args_need_initialize.append(name)
            return default

        # Initialize args (if needed)
        if need_initialize and name in self._args_need_initialize:
            return default

        return self._get(name if not other_names else [name] + other_names)

    def _register(self, name: str,
                  default: Any,
                  argtype: str,
                  short_name: str | None = None,
                  other_names: list[str] | None = None,
                  desc_in_model_summary: str | tuple[str, str] | None = None):
        """
        Register a new arg.
        """
        if not name in self._arg_list:
            self._arg_list.append(name)
            self._arg_type[name] = argtype
            self._args_default[name] = default

            if short_name:
                self._arg_short_name[short_name] = name

            if other_names is not None and len(other_names):
                self._arg_aliases[name] = other_names

            if d := desc_in_model_summary:
                self._args_desc[name] = d

    def _get(self, name: str | list[str]):
        """
        Get value of a arg.

        :param name: The name (or all other names) of the arg. \
            NOTE: `name[0]` should be the main name.       
        :param default: default value of the arg
        :param argtype: type of the arg, can be
            - `STATIC`
            - `DYNAMIC`
            - `TEMPORARY`
            - ...
        """

        # arg dict index:
        # _args_load: 0
        # _args_running: 1
        # _args_manually: 99
        # _args_default: -1

        names = name if isinstance(name, list) else [name]
        argtype = self._arg_type[names[0]]
        default = self._args_default[names[0]]

        if argtype == STATIC:
            order = [99, 0, 1, -1]
        elif argtype == DYNAMIC:
            order = [99, 1, 0, -1]
        elif argtype == TEMPORARY:
            order = [99, 1, -1]
        else:
            raise ValueError('Wrong arg type.')

        # Get args from all dictionaries.
        value = None
        for index in order:
            value = self._get_args_by_index_and_name(index, names)

            if value is not None:
                break
            else:
                continue

        if value is None:
            value = default

        value = type(default)(value)

        return value
