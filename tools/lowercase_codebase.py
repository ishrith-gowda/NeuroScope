#!/usr/bin/env python3
"""
bulk lowercase enforcer for all comments, docstrings, and print statements.

processes all .py files in the neuroscope project and converts:
- comments (# ...) to lowercase
- docstrings (triple-quoted strings) to lowercase
- print() string arguments to lowercase

preserves:
- code identifiers, class names, function names
- string literals that are not docstrings or print args
- file paths, urls, and format strings
- section dividers (=== or ---)
"""

import os
import re
import sys
from pathlib import Path


def lowercase_comments(line: str) -> str:
    """lowercase the comment portion of a line."""
    # find inline comment (not inside a string)
    # simple approach: find # that's not inside quotes
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == '#' and not in_single and not in_double:
            # found comment start
            code_part = line[:i]
            comment_part = line[i:]
            # lowercase the comment text but preserve # and spacing
            if comment_part.startswith('#'):
                # preserve section dividers like # ====== or # ------
                stripped = comment_part.lstrip('#').lstrip()
                if stripped and all(c in '=-~*' for c in stripped.rstrip()):
                    return line
                # preserve shebang
                if comment_part.startswith('#!'):
                    return line
                # preserve encoding declarations
                if 'coding' in comment_part and (':' in comment_part or '=' in comment_part):
                    return line
                # preserve noqa, type: ignore, fmt: directives
                if any(d in comment_part.lower() for d in ['noqa', 'type: ignore', 'fmt:', 'pragma:', 'pylint:']):
                    return line
                return code_part + comment_part.lower()
            return line
    return line


def lowercase_docstring(text: str) -> str:
    """lowercase a docstring while preserving structure."""
    # preserve section dividers
    lines = text.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        # preserve lines that are just dividers
        if stripped and all(c in '=-~*' for c in stripped):
            result.append(line)
        else:
            result.append(line.lower())
    return '\n'.join(result)


def process_file(filepath: str) -> bool:
    """process a single python file. returns true if modified."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (UnicodeDecodeError, IOError):
        return False

    original = content

    # step 1: lowercase docstrings (triple-quoted strings at start of module/class/function)
    # match triple-quoted strings
    def replace_docstring(match):
        quote = match.group(1)  # """ or '''
        body = match.group(2)
        return quote + lowercase_docstring(body) + quote

    # handle """ docstrings
    content = re.sub(
        r'(""")(.+?)(""")',
        lambda m: m.group(1) + lowercase_docstring(m.group(2)) + m.group(3),
        content,
        flags=re.DOTALL,
    )

    # handle ''' docstrings
    content = re.sub(
        r"(''')(.+?)(''')",
        lambda m: m.group(1) + lowercase_docstring(m.group(2)) + m.group(3),
        content,
        flags=re.DOTALL,
    )

    # step 2: lowercase comments
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        new_lines.append(lowercase_comments(line))
    content = '\n'.join(new_lines)

    # step 3: lowercase print() string arguments
    # match print("...") and print(f"...") patterns
    def lowercase_print_str(match):
        prefix = match.group(1)  # print( or print(f
        quote = match.group(2)   # " or '
        body = match.group(3)    # string content
        end_quote = match.group(4)
        # preserve f-string expressions {var_name}
        def lower_outside_braces(s):
            result = []
            depth = 0
            current = []
            for ch in s:
                if ch == '{':
                    if depth == 0 and current:
                        result.append(''.join(current).lower())
                        current = []
                    depth += 1
                    current.append(ch)
                elif ch == '}':
                    current.append(ch)
                    depth -= 1
                    if depth == 0:
                        result.append(''.join(current))
                        current = []
                else:
                    current.append(ch)
            if current:
                if depth > 0:
                    result.append(''.join(current))
                else:
                    result.append(''.join(current).lower())
            return ''.join(result)

        lowered = lower_outside_braces(body)
        return prefix + quote + lowered + end_quote

    # print with double quotes
    content = re.sub(
        r'(print\(f?)(")([^"]*?)(")',
        lowercase_print_str,
        content,
    )
    # print with single quotes
    content = re.sub(
        r"(print\(f?)(')([^']*?)(')",
        lowercase_print_str,
        content,
    )

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    project_root = Path(__file__).parent.parent

    # directories to process
    dirs_to_process = [
        project_root / "neuroscope",
        project_root / "journal_extension",
        project_root / "scripts",
        project_root / "tools",
        project_root / "unified",
    ]

    modified_count = 0
    total_count = 0

    for search_dir in dirs_to_process:
        if not search_dir.exists():
            continue
        for py_file in sorted(search_dir.rglob("*.py")):
            # skip venv
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
            total_count += 1
            if process_file(str(py_file)):
                modified_count += 1
                print(f"  modified: {py_file.relative_to(project_root)}")

    # also process setup.py at root
    setup_py = project_root / "setup.py"
    if setup_py.exists():
        total_count += 1
        if process_file(str(setup_py)):
            modified_count += 1
            print(f"  modified: setup.py")

    print(f"\nprocessed {total_count} files, modified {modified_count}")


if __name__ == "__main__":
    main()
