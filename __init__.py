"""
Q-Pid
---
A training "engine" for trajectory prediction models (based on PyTorch).
"""

from . import (applications, args, base, help, model, silverballers, training,
               utils)
from .help import print_help_info, register_new_args

__SYSTEM_MANAGER = base.BaseManager(name='SYSTEM')


def log(s: str, level: str = 'info', raiseError=None):
    """
    The system-level log function.

    :param s: The text to log.
    :param level: Log level, can be one of `['info', 'warning', 'error', 'debug']`.
    :param raiseError: Some exception to raise after logging.
    """
    __SYSTEM_MANAGER.log(s, level, raiseError)


def _log_mod_loaded(name: str):
    log(f'Mod `{name}` successfully loaded.')


def set_log_path(p: str):
    """
    Set the path of the log file.
    """
    utils.LOG_FILE = p


def set_log_stream_handler(handler):
    """
    Set the log handler (which handles terminal-like outputs).
    Type of the handler should be `logging.Handler`.
    """
    utils.LOG_STREAM_HANDLER = handler
