"""
Q-Pid
---
A training "engine" for trajectory prediction models (based on TensorFlow 2).
"""

from . import (applications, args, base, managers, model, silverballers,
               training, utils)
from .__getHelp import print_help_info


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
