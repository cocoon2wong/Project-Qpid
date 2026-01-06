"""
@Author: Conghao Wong
@Date: 2023-09-06 19:26:52
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-06 15:53:38
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import inspect
import re
from typing import TypeVar

from .args import Args, EmptyArgs

TArgs = TypeVar('TArgs', bound=EmptyArgs)

FLAG = '<!-- DO NOT CHANGE THIS LINE -->'
TARGET_FILE = './README.md'
MAX_SPACE = 20  # Only used for "basic display" mode

ARGS_DIC: dict = {
    Args: ['Basic Args', None],
}

# -------------------------
# Templates
# -------------------------

# Basic display (pure Markdown list, doc is squashed into one line)
DOC_TEMPLATE_REGULAR = """- `--{name}`{short}:
  {doc}
    - Type=`{dtype}`, argtype=`{argtype}`;{aliases}
    - The default value is `{default}`.
"""

# HTML/details shell + Markdown inside (kramdown-friendly)
# - markdown="1" lets kramdown parse inner markdown within <details>
# - summary markdown="span" keeps it inline
DOC_TEMPLATE_DETAILS_MD = """
<details markdown="1">
<summary markdown="span"><code>--{name}</code>{short}</summary>

{doc}

- Type=`{dtype}`, argtype=`{argtype}`{aliases}
- The default value is `{default}`.

</details>
"""


# -------------------------
# Doc processing
# -------------------------

def _squash_to_one_line(doc: str) -> str:
    """
    Convert doc to a single line (for basic display mode).
    Keeps semantics simple and stable.
    """
    # Replace newlines with spaces and squeeze repeated spaces
    doc = doc.replace('\n', ' ')
    for _ in range(MAX_SPACE):
        if '  ' not in doc:
            break
        doc = doc.replace('  ', ' ')

    doc = doc.strip()

    if not doc:
        doc = '(Working in process)'

    # Make it look like a sentence if it's plain text
    if not doc.endswith(('.', '!', '?', '。', '！', '？')):
        doc += '.'
    return doc


def _keep_markdown(doc: str | None) -> str:
    """
    Keep doc as Markdown (for details/html mode), only clean indentation
    and remove leading/trailing blank lines.
    """
    if not doc:
        return '(Working in process)'

    # cleandoc: dedent + trim leading/trailing blank lines
    doc = inspect.cleandoc(doc)
    doc = re.sub(r'(?<!\n)\n(?!\n)', ' ', doc)

    # If it's just a single line of plain text, add a period (optional)
    if '\n' not in doc and not doc.endswith(('.', '!', '?', ':', '。', '！', '？', '：')):
        doc += '.'

    return doc


def _format_short_names(arg: Args, name: str) -> str:
    short_name_desc = ''
    if name in arg._arg_short_name.values():
        short_names = [k for k, v in arg._arg_short_name.items() if v == name]
        if short_names:
            ss = ' '.join([f'<code>-{s}</code>' for s in short_names])
            short_name_desc = f' (short for {ss})'
    return short_name_desc


def _format_aliases(arg: Args, name: str, html: bool) -> str:
    """
    In basic display: keep your original extra bullet line.
    In details/html: keep it inline on the Type line ("; also: ...").
    """
    if name not in arg._arg_aliases:
        return ''

    aliases = [f'`--{a}`' for a in arg._arg_aliases[name]]

    if not html:
        return (
            '\n    - This arg can also be spelled as '
            + ', '.join(aliases)
            + ';'
        )
    else:
        return '; also: ' + ', '.join(aliases)


# -------------------------
# Main public APIs
# -------------------------

def get_arg_doc(arg: Args, html: bool = False) -> list[str]:
    """
    Get documents of all arg-elements for the given `Args` object.

    - html=False: basic markdown list; doc is squashed into one line
    - html=True:  <details> shell with markdown inside (kramdown-friendly)
    """
    results: list[str] = []

    for name in arg._arg_list:
        default = arg._args_default[name]
        dtype = type(default).__name__
        argtype = arg._arg_type[name]

        short_name_desc = _format_short_names(arg, name)
        aliases_desc = _format_aliases(arg, name, html=html)

        raw_doc = getattr(arg.__class__, name).__doc__

        if not html:
            doc = _squash_to_one_line(raw_doc or '')
            template = DOC_TEMPLATE_REGULAR
            line = template.format(
                name=name,
                short=short_name_desc,
                doc=doc,
                dtype=dtype,
                argtype=argtype,
                aliases=aliases_desc,
                default=default,
            )
        else:
            # keep markdown (multi-line) inside details
            doc = _keep_markdown(raw_doc)
            template = DOC_TEMPLATE_DETAILS_MD
            line = template.format(
                name=name,
                short=short_name_desc,
                doc=doc,
                dtype=dtype,
                argtype=argtype,
                aliases=aliases_desc,
                default=default,
            )

            # NOTE: We intentionally DO NOT convert backticks to <code> here.
            # Let kramdown/GFM render markdown naturally.
            # (We already wrap `--name` with <code> in <summary> for stability.)

        results.append(line)

    return results


def get_args_docs(args: list[Args], titles: list[str], html: bool = False) -> list[str]:
    """
    Get documents from the given `Args` objects.

    Dedup logic remains:
    - args[0] is the "base" set; later args won't re-emit same arg names.
    """
    lines: list[str] = []
    processed_args: list[list[str]] = [[] for _ in range(len(args))]

    for index, (arg, title) in enumerate(zip(args, titles)):
        lines += [f'\n### {title}\n\n']
        c = get_arg_doc(arg, html=html)
        c.sort()

        for new_line in c:
            if not html:
                # - `--xxx`...
                # grab between first pair of backticks
                name = new_line.split('`')[1]
            else:
                # <summary ...><code>--xxx</code>...
                m = re.search(r'<code>--(.+?)</code>', new_line)
                if not m:
                    # fallback: try markdown `--xxx` if user edits template
                    m2 = re.search(r'`--(.+?)`', new_line)
                    name = m2.group(1) if m2 else new_line
                else:
                    name = m.group(1)

            processed_args[index].append(name)
            if (name not in processed_args[0]) or (index == 0):
                lines.append(new_line)

    return lines


def update_readme(new_lines: list[str], md_file: str):
    with open(md_file, 'r', encoding='utf-8') as f:
        lines = f.read()

    try:
        pattern = re.findall(
            rf'([\s\S]*)({re.escape(FLAG)})([\s\S]*)({re.escape(FLAG)})([\s\S]*)',
            lines,
        )[0]
        all_lines = list(pattern[:2]) + new_lines + list(pattern[-2:])
    except Exception:
        flag_line = f'{FLAG}\n'
        all_lines = [lines, flag_line] + new_lines + [flag_line]

    with open(md_file, 'w+', encoding='utf-8') as f:
        f.writelines(all_lines)
