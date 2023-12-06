"""
@Author: Conghao Wong
@Date: 2022-11-11 10:05:11
@LastEditors: Conghao Wong
@LastEditTime: 2023-12-06 15:54:59
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import logging
from typing import Any, TypeVar

import torch
from tqdm import tqdm

from .. import utils

T = TypeVar('T')


class BaseObject():
    """
    Basic object with logging-related functions.

    Public Methods
    --------------
    ### Log and print methods
    ```
    # log information
    (method) log: (self: Self@BaseObject, s: str, level: str = 'info') -> str

    # print parameters with the format
    (method) print_parameters: (self: Self@BaseObject, 
                                title: str = 'null',
                                **kwargs: Any) -> None

    # print information about the manager object itself
    (method) print_info: (self: Self@BaseObject, **kwargs: Any) -> None

    # print information of the manager and its members
    (method) print_info_all: (self: Self@BaseObject,
                              include_self: bool = True) -> None

    # timebar
    (method) log_timebar: (inputs, text='', return_enumerate=True) -> (enumerate | tqdm)
    ```
    """

    def __init__(self, name: str | None = None):
        super().__init__()

        try:
            self.name = name
        except AttributeError:
            pass

        # create or restore a logger
        logger = logging.getLogger(name=f'`{name}` ({type(self).__name__})')
        logger.propagate = False

        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)

            # Add the file handler (`./test.log`)
            fhandler = logging.FileHandler(filename=utils.LOG_FILE, mode='a')
            fhandler.setLevel(logging.INFO)

            # Add the terminal handler
            thandler = utils.LOG_STREAM_HANDLER
            thandler.setLevel(logging.INFO)

            # Add formatters
            # Files
            fformatter = logging.Formatter(
                '[%(asctime)s][%(levelname)s] %(name)s: %(message)s')
            fhandler.setFormatter(fformatter)

            # Terminal
            tformatter = logging.Formatter(
                '[%(levelname)s] %(name)s: %(message)s')
            thandler.setFormatter(tformatter)

            logger.addHandler(fhandler)
            logger.addHandler(thandler)

        self.logger = logger
        self.bar: tqdm | None = None

    def log(self, s: str, level: str = 'info',
            raiseError: type[BaseException] | None = None):
        """
        Log information to files and console.

        :param s: The text to log.
        :param level: Log level, can be one of `['info', 'warning', 'error', 'debug']`.
        :param raiseError: Some exception to raise after logging.
        """
        if level == 'info':
            self.logger.info(s)

        elif level == 'warning':
            self.logger.warning(s)

        elif level == 'error':
            self.logger.error(s)

        elif level == 'debug':
            self.logger.debug(s)

        else:
            raise NotImplementedError

        if raiseError:
            raise raiseError(s)

        return s

    def timebar(self, inputs: T, text='') -> T:
        bf = ('{desc}: {percentage:3.0f}%|' +
              '{bar}' +
              '| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, ' +
              '{rate_fmt}{postfix}]')
        self.bar = tqdm(inputs, desc=text, bar_format=bf)
        return self.bar

    def update_timebar(self, item: str | torch.Tensor | dict[str, Any],
                       pos='end'):
        """
        Update the tqdm time bar.

        :param item: The string or dictionary to show on the time bar.
        :param pos: Position to show, can be `'end'` or `'start'`.
        """
        if not self.bar:
            raise ValueError(f'`{self}` object has no time bars!')

        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    if value.requires_grad:
                        value = value.detach()

                    item[key] = value.cpu().numpy()

        elif isinstance(item, torch.Tensor):
            item = item.cpu().numpy()

        if pos == 'end':
            if type(item) is str:
                self.bar.set_postfix_str(item)
            elif type(item) is dict:
                self.bar.set_postfix(item)
            else:
                raise ValueError(item)

        elif ((pos == 'start') and (isinstance(item, str))):
            self.bar.set_description(item)
        else:
            raise NotImplementedError(pos)

    def print_info(self, **kwargs):
        """
        Print information about the object itself.
        """
        self.print_parameters(**kwargs)

    def print_parameters(self, title='null', **kwargs):
        if title == 'null':
            title = ''

        print(f'>>> [{self.name}]: {title}')
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.numpy()

            elif isinstance(value, list | tuple):
                if len(value) > utils.LOG_MAX_LIST_LEN:
                    value = list(value[:utils.LOG_MAX_LIST_LEN]) + ['...']

            if value is not None:
                print(f'    - {key}: {value}.')
            else:
                print(f'    + {key}:')

        print('')

    @staticmethod
    def log_bar(percent, total_length=30):

        bar = (''.join('=' * (int(percent * total_length) - 1))
               + '>')
        return bar
