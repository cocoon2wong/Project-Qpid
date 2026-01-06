"""
Q-Pid
---
A training "engine" for trajectory prediction models (based on PyTorch).
"""

from . import applications, args, base, help, model, training, utils
from .__system import sys_mgr
from .cli import entrance

add_arg_alias = sys_mgr.add_arg_alias
get_structure = sys_mgr.get_structure
get_model = sys_mgr.get_model

register = sys_mgr.register
register_args = sys_mgr.register_args
register_input_type = sys_mgr.register_input_type

log = sys_mgr.log
set_log_path = sys_mgr.set_log_path
set_log_stream_handler = sys_mgr.set_log_stream_handler

get_all_args_docs = sys_mgr.get_all_args_docs


# Register simple trajectory prediction models
register(
    linear=[applications.Linear, None],
    static=[applications.Static, None],
    as_static_models=True,
)

# Register basic prediction args
register_args(args.Args, 'Basic Args')
