"""
Q-Pid
---
A training "engine" for trajectory prediction models (based on TensorFlow 2).
"""

from . import (applications, args, base, help, model, silverballers, training,
               utils)
from .help import print_help_info, update_args_dic


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
