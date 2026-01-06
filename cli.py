"""
@Author: Conghao Wong
@Date: 2026-01-06 18:27:21
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-06 18:50:36
@Github: https://cocoon2wong.github.io
@Copyright 2026 Conghao Wong, All Rights Reserved.
"""

import importlib
import sys
from pathlib import Path

from . import args
from .__system import sys_mgr
from .utils import EXCLUDE_PKGS


def entrance_cli():
    entrance(sys.argv, train_or_test=True)


def entrance(terminal_args: list[str], train_or_test=True):
    """
    Main entrance for training or testing models from terminals.
    """
    # Import all current models
    import_models()

    _temp_args = args.Args(terminal_args, is_temporary=True)

    # Check if `-h` or `--help` in args
    if (h := _temp_args.help) != 'null':
        sys_mgr.get_all_args_docs('all_args' if h == 'True' else h)
        exit()

    # Init the structure
    t_type = sys_mgr.get_structure(_temp_args.model)
    t = t_type(terminal_args)

    # Start training or testing
    if train_or_test:
        t.train_or_test()

    # (DEBUG) Verbose mode
    if t.args.verbose:
        t.print_info_all()

    return t


def import_models():
    qpid_dir = Path(__file__).resolve().parent      # .../qpid
    root = qpid_dir.parent                          # repo_root

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    for path in root.iterdir():
        if not path.is_dir():
            continue
        if not (path / '__init__.py').exists():
            continue

        name = path.name
        if name in EXCLUDE_PKGS:
            continue

        try:
            importlib.import_module(name)
            sys_mgr.log(f'Plugin `{name}` loaded.')
        except Exception as e:
            sys_mgr.log(f'Plugin `{name}` failed to load: {e}',
                        level='warning')
