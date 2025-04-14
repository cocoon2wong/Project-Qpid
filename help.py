"""
@Author: Conghao Wong
@Date: 2023-09-06 19:26:52
@LastEditors: Conghao Wong
@LastEditTime: 2025-04-14 17:51:29
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

DOC_TEMPLATE_REGULAR = """- `--{}`{}:
  {}
    - Type=`{}`, argtype=`{}`;{}
    - The default value is `{}`.
"""

DOC_TEMPLATE_HTML = """
<details>
    <summary>
        `--{}`{}
    </summary>
    <p>
        {}
    </p>
    <ul>
        <li>Type=`{}`, argtype=`{}`;</li>{}
        <li>The default value is `{}`.</li>
    </ul>
</details>
"""


def get_arg_doc(arg: Args, html=False) -> list[str]:
    """
    Get documents of all arg-elements for the given `Args` object.
    """

    results = []
    for name in arg._arg_list:

        default = arg._args_default[name]
        dtype = type(default).__name__
        argtype = arg._arg_type[name]

        # Check if there are any short names
        short_name_desc = ''
        if name in arg._arg_short_name.values():
            short_names = []
            for key in arg._arg_short_name.keys():
                if arg._arg_short_name[key] == name:
                    short_names.append(key)

            ss = ' '.join(['`-{}`'.format(s) for s in short_names])
            short_name_desc = f' (short for {ss})'

        # Check if there are any aliases
        other_names_desc = ''
        if name in arg._arg_aliases.keys():
            aliases = [f'`--{a}`' for a in arg._arg_aliases[name]]

            if not html:
                other_names_desc = (
                    '\n    - This arg can also be spelled as '
                    + ', '.join(aliases)
                    + ';'
                )
            else:
                other_names_desc = (
                    '<li>This arg can also be spelled as'
                    + ', '.join(aliases)
                    + ';</li>'
                )

        doc = getattr(arg.__class__, name).__doc__

        if doc is None:
            doc = '(Working in process)'

        # Trim spaces (as tabs) and `\n`s
        doc = doc.replace('\n', ' ')
        for _ in range(MAX_SPACE):
            if '  ' not in doc:
                break
            doc = doc.replace('  ', ' ')

        while doc.startswith(' '):
            doc = doc[1:]

        while doc.endswith(' '):
            doc = doc[:-1]

        if not doc.endswith('.'):
            doc += '.'

        # Fill into templates
        t = DOC_TEMPLATE_REGULAR if not html else DOC_TEMPLATE_HTML

        line = t.format(
            name, short_name_desc,
            doc, dtype, argtype,
            other_names_desc, default,
        )

        # Post-fix
        if html:
            line = re.sub('`(.+?)`', '<code>\\1</code>', line)

        results.append(line)

    return results


def get_args_docs(args: list[Args], titles: list[str], html=False) -> list[str]:
    """
    Get documents from the given `Args` objects.
    """

    lines = []
    processed_args = [[] for _ in range(len(args))]

    for index, (arg, title) in enumerate(zip(args, titles)):
        lines += [f'\n### {title}\n\n']
        c = get_arg_doc(arg, html=html)
        c.sort()

        for new_line in c:
            if not html:
                name = new_line.split('`')[1]
            else:
                name = re.findall('<code>(.+?)</code>', new_line)[0]

            processed_args[index].append(name)
            if (name not in processed_args[0]) or (index == 0):
                lines.append(new_line)

    return lines


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
