"""
@Author: Conghao Wong
@Date: 2022-07-27 20:47:50
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 13:34:53
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from . import interpHandlers
from .__baseSubnetwork import BaseSubnetwork, BaseSubnetworkStructure


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
    def get_structure(cls, name: str) -> type[BaseSubnetworkStructure]:
        """
        Get structure type of the given model.
        """
        return cls.__get(name)[0]

    @classmethod
    def get_model(cls, name: str) -> type[BaseSubnetwork]:
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
