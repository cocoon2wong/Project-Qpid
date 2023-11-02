"""
@Author: Conghao Wong
@Date: 2022-07-27 20:47:50
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-02 18:18:59
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import TypeVar, overload

from . import interpHandlers
from .__baseSubnetwork import BaseSubnetwork, BaseSubnetworkStructure

TStructure = TypeVar('TStructure', bound=BaseSubnetworkStructure)
TModel = TypeVar('TModel', bound=BaseSubnetwork)


class SILVERBALLERS_DICT():
    """
    Silverballers Dictionary
    ---
    A class that manages all silverballers structures and models.

    ### Public Methods

    - Model register:

        ```python
        SILVERBALLERS_DICT.register(**kwargs)
        ```

    - Get model type (class name):

        ```python
        SILVERBALLERS_DICT.get_model(name)
        ```

    - Get model's training structure type (class name):

        ```python
        SILVERBALLERS_DICT.get_structure(name)
        ```
    """

    __S_DICT: dict[str, tuple[type[BaseSubnetworkStructure],
                              type[BaseSubnetwork]]] = {}
    __S_DICT.update(dict(

    ))

    # Interpolation Handlers
    __S_DICT.update(interpHandlers.INTERPOLATION_HANDLER_DICT)

    @classmethod
    def register(cls, **kwargs):
        """
        Register new silverballers models.
        Arg format:
        `MODEL_NAME = [STRUCTURE_TYPE, MODEL_TYPE]`

        For example, you can register vertical models by
        ```python
        SILVERBALLERS_DICT.register(va=[VA, VAModel],
                                    vb=[VB, VBModel])
        ```
        """
        for k, v in kwargs.items():
            cls.__S_DICT[k] = v

    @classmethod
    @overload
    def get_structure(cls, name: str) -> type[BaseSubnetworkStructure]: ...

    @classmethod
    @overload
    def get_structure(
        cls, name: str, stype: type[TStructure] | None = None) -> type[TStructure]: ...

    @classmethod
    def get_structure(cls, name: str, stype=None):
        """
        Get structure type of the given model.

        :param name: Name of the structure.
        :param stype: Type of the structure. It is only used for type-hinting.
        """
        return cls.__get(name)[0]

    @classmethod
    @overload
    def get_model(cls, name: str) -> type[BaseSubnetwork]: ...

    @classmethod
    @overload
    def get_model(cls, name: str,
                  mtype: type[TModel] | None = None) -> type[TModel]: ...

    @classmethod
    def get_model(cls, name: str, mtype=None):
        """
        Get model type of the given model.
        """
        return cls.__get(name)[1]

    @classmethod
    def __get(cls, name: str):
        if not name in cls.__S_DICT.keys():
            raise NotImplementedError(
                f'Model type `{name}` is not supported.')

        return cls.__S_DICT[name]
