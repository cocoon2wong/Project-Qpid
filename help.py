"""
@Author: Conghao Wong
@Date: 2023-09-06 19:26:52
@LastEditors: Conghao Wong
@LastEditTime: 2024-03-20 20:29:48
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import re
from typing import TypeVar

from .args import Args, EmptyArgs

TArgs = TypeVar('TArgs', bound=EmptyArgs)

FLAG = '<!-- DO NOT CHANGE THIS LINE -->'
TARGET_FILE = './README.md'
MAX_SPACE = 20

ARGS_DIC: dict = {
    Args: ['Basic Args', None],
}


def read_comments(args: Args) -> list[str]:

    results = []
    for arg in args._arg_list:

        name = arg
        default = args._args_default[name]
        dtype = type(default).__name__
        argtype = args._arg_type[name]

        short_name_desc = ''
        if name in args._arg_short_name.values():
            short_names = []
            for key in args._arg_short_name.keys():
                if args._arg_short_name[key] == name:
                    short_names.append(key)

            ss = ' '.join(['`-{}`'.format(s) for s in short_names])
            short_name_desc = f' (short for {ss})'

        doc = getattr(args.__class__, arg).__doc__
        doc = doc.replace('\n', ' ')
        for _ in range(MAX_SPACE):
            doc = doc.replace('  ', ' ')

        s = (f'- `--{name}`' + short_name_desc +
             f': type=`{dtype}`, argtype=`{argtype}`.\n' +
             f' {doc}\n  The default value is `{default}`.')
        results.append(s + '\n')
        # print(s)

    return results


def get_doc(args: list[Args], titles: list[str],
            father_indices: list[int]) -> list[str]:

    new_lines = []
    all_args = [[] for _ in range(len(args))]

    for index, (arg, title, father) in enumerate(zip(args, titles, father_indices)):
        new_lines += [f'\n### {title}\n\n']
        c = read_comments(arg)
        c.sort()

        for new_line in c:
            name = new_line.split('`')[1]
            all_args[index].append(name)
            if father is None:
                father = 0
            if (name not in all_args[father]) or (index == father):
                new_lines.append(new_line)

    return new_lines


def update_readme(new_lines: list[str], md_file: str):
    with open(md_file, 'r') as f:
        lines = f.readlines()
    lines = ''.join(lines)

    try:
        pattern = re.findall(
            f'([\s\S]*)({FLAG})([\s\S]*)({FLAG})([\s\S]*)', lines)[0]
        all_lines = list(pattern[:2]) + new_lines + list(pattern[-2:])

    except:
        flag_line = f'{FLAG}\n'
        all_lines = [lines, flag_line] + new_lines + [flag_line]

    with open(md_file, 'w+') as f:
        f.writelines(all_lines)


def print_help_info(value: str | None = None):
    files = []
    for T in ARGS_DIC.keys():
        ignore_flag = T._ignore_value_check
        T._ignore_value_check = True
        _arg = T(is_temporary=True)
        files.append(_arg)
        T._ignore_value_check = ignore_flag

    titles = [v[0] for v in ARGS_DIC.values()]
    father_indices = [v[1] for v in ARGS_DIC.values()]

    doc_lines = get_doc(files, titles, father_indices)
    if value is None:
        pass
    elif value == 'all_args':
        [print(doc) for doc in doc_lines]
    else:
        doc_lines = [doc for doc in doc_lines if doc[5:].startswith(value)]
        [print(doc) for doc in doc_lines]
    return doc_lines


def register_new_args(arg_type: type[TArgs],
                      friendly_name: str,
                      package_name: str | None = None,
                      farther_index: int | None = None):
    """
    Register new args defined by additional mods to the training structure.

    :param arg_type: Type of the new arg object.
    :param friendly_name: Friendly name of the class of args that showed to users.
    :param package_name: Name of the package where the new args are included.
    """
    if not package_name:
        package_name = arg_type.__name__

    ARGS_DIC.update({arg_type: [friendly_name, farther_index]})
