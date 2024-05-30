"""
Q-Pid
---
A training "engine" for trajectory prediction models (based on PyTorch).
"""

from . import applications, args, base, help, model, training, utils
from .__system import SystemManager as __SysMgr

sys_mgr = __SysMgr()

add_arg_alias = sys_mgr.add_arg_alias
get_structure = sys_mgr.get_structure
get_model = sys_mgr.get_model

register = sys_mgr.register
register_args = sys_mgr.register_args
register_input_type = sys_mgr.register_input_type

log = sys_mgr.log
set_log_path = sys_mgr.set_log_path
set_log_stream_handler = sys_mgr.set_log_stream_handler
print_help_info = sys_mgr.print_help_info


register(
    linear=[applications.Linear, None],
    static=[applications.Static, None],
)

register_args(args.Args, 'Basic Args')
