"""
@Author: Conghao Wong
@Date: 2024-05-30 09:58:23
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 13:29:39
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

from typing import Callable, TypeVar, overload

from . import args, constant, utils
from .args import Args, EmptyArgs
from .base import BaseManager
from .dataset import AgentManager
from .help import get_doc
from .model import Model
from .training import Structure

TStructure = TypeVar('TStructure', bound=Structure)
TModel = TypeVar('TModel', bound=Model)


class SystemManager(BaseManager):
    """
    Qpid System Manager
    ---
    A class that manages all qpid structures and models.

    ### Public Methods

    - Model register:

        ```python
        register(**kwargs)
        ```

    - Arg register:

        ```python
        register_args(atype, friendly_name)
        ```

    - Get model type (class name):

        ```python
        get_model(name)
        ```

    - Get model's training structure type (class name):

        ```python
        get_structure(name)
        ```
    """

    def __init__(self):
        super().__init__(name='SYSTEM')

        self.__model_dict: dict[str, tuple[type[Structure],
                                           type[Model]]] = {}
        self.__args_dict: dict[type[Args | EmptyArgs], str] = {}
        self.__input_type_handler_dict: dict = {}

    @property
    def log_path(self) -> str:
        return utils.LOG_FILE

    @property
    def log_stream_handler(self):
        return utils.LOG_STREAM_HANDLER

    @property
    def model_dict(self):
        return self.__model_dict

    @property
    def args_dict(self):
        return self.__args_dict

    @property
    def input_type_handler_dict(self) -> \
            dict[str, Callable[[AgentManager], None] | None]:
        return self.__input_type_handler_dict

    @property
    def command_alias(self):
        return args.ARG_ALIAS

    def set_log_path(self, p: str):
        """
        Set the path of the log file.
        """
        utils.LOG_FILE = p

    def set_log_stream_handler(self, handler):
        """
        Set the log handler (which handles terminal-like outputs).
        Type of the handler should be `logging.Handler`.
        """
        utils.LOG_STREAM_HANDLER = handler

    def register(self, **kwargs):
        """
        Register new qpid models.
        Arg format:
        `MODEL_NAME = [STRUCTURE_TYPE, MODEL_TYPE]`

        For example, you can register vertical models by
        ```python
        register(va=[VA, VAModel], vb=[VB, VBModel])
        ```
        """
        for k, v in kwargs.items():
            self.__model_dict[k] = v

    def register_args(self, atype: type[Args | EmptyArgs],
                      friendly_name: str):
        self.__args_dict[atype] = friendly_name

    def register_input_type(self, name: str,
                            handler: Callable[[AgentManager], None] | None = None):
        setattr(constant.INPUT_TYPES, name, name)
        self.__input_type_handler_dict[name] = handler

    def add_arg_alias(self, alias: str | list[str],
                      command: list[str],
                      pattern: str | None = None):
        """
        Add a new alias for running args.

        :param alias: The alias string(s).
        :param command: Commands, should be a list of strings.
        :param pattern: Patterns to expand the received arg-value.
        """
        if isinstance(alias, str):
            alias = [alias]

        for a in alias:
            if pattern:
                args.ARG_ALIAS[a] = (command, pattern)
            else:
                args.ARG_ALIAS[a] = command

    def print_help_info(self, value: str | None = None):
        files = []
        for T in self.args_dict.keys():
            ignore_flag = T._ignore_value_check
            T._ignore_value_check = True
            _arg = T(is_temporary=True)
            files.append(_arg)
            T._ignore_value_check = ignore_flag

        titles = list(self.args_dict.values())
        doc_lines = get_doc(files, titles)

        if value is None:
            pass
        elif value == 'all_args':
            [print(doc) for doc in doc_lines]
        else:
            doc_lines = [doc for doc in doc_lines if doc[5:].startswith(value)]
            [print(doc) for doc in doc_lines]

        return doc_lines

    @overload
    def get_structure(self, name: str) -> type[Structure]: ...

    @overload
    def get_structure(self, name: str,
                      stype: type[TStructure] | None = None) -> type[TStructure]: ...

    def get_structure(self, name: str, stype=None):
        """
        Get the structure type of the given model (str).

        :param name: Name of the structure.
        :param stype: Type of the structure. It is only used for type-hinting.
        """
        if name == 'MKII':
            import qpid.mods.silverballers

        return self.__get(name)[0]

    @overload
    def get_model(self, name: str) -> type[Model]: ...

    @overload
    def get_model(self, name: str,
                  mtype: type[TModel] | None = None) -> type[TModel]: ...

    def get_model(self, name: str, mtype=None):
        """
        Get the model type of the given model (str).
        """
        return self.__get(name)[1]

    def __get(self, name: str):
        if not name in self.__model_dict.keys():
            raise NotImplementedError(
                f'Model type `{name}` is not supported.')

        return self.__model_dict[name]
