"""
## shlex for all

The standard lib `shlex` provides parsing only for Unix, this library is
intended to provide such parsing for all platforms. You can use `join`,
`split` and `quote` function. Please remember that join right after split
may not give the original string back.

On Windows, the parsing follows documents on
https://docs.python.org/3.11/library/subprocess.html#converting-an-argument-sequence-to-a-string-on-windows
and
https://learn.microsoft.com/en-us/windows/win32/api/shellapi/nf-shellapi-commandlinetoargvw#remarks

On other platforms, the functions are same as `shlex`.
"""

import re
import shlex
import sys
from typing import Iterable

__all__ = ["join", "quote", "split", "join_unix", "quote_unix", "split_unix", "join_win", "quote_win", "split_win"]

join_unix = shlex.join
quote_unix = shlex.quote
split_unix = shlex.split


def quote_win(s: str):
    if type(s) != str:
        raise TypeError("s must be `str`")
    if " " not in s and "\t" not in s and '"' not in s:
        return s
    ret = s
    if '"' in ret:
        ret = ret.replace('"', '\\"')
    if " " in ret or "\t" in ret:
        ret = '"' + ret + '"'
    ret = re.sub('(\\\\+)\\\\"', '\\1\\1\\"', ret)
    return ret


def join_win(s: Iterable):
    return " ".join(quote_win(i) for i in s)


def split_win(s: str):
    if type(s) != str:
        raise TypeError("s must be `str`")
    s = s.strip(" \t")
    if not s:
        return []
    ret = []
    i = 0
    in_quote = False
    next_str = ""
    while i < len(s):
        if s[i] in " \t":
            if in_quote:
                next_str += s[i]
                i += 1
            else:
                if s[i - 1] not in " \t":
                    ret.append(next_str)
                    next_str = ""
                while i < len(s) and s[i] in " \t":
                    i += 1
        elif s[i] == "\\":
            backslashs_and_next = re.match(r"(\\+)(.|$)", s[i:])
            backslashs = len(backslashs_and_next.group(1))
            if backslashs_and_next.group(2) != '"':
                i += backslashs
                next_str += "\\" * backslashs
            elif backslashs % 2 == 0:
                i += backslashs + 1
                next_str += "\\" * (backslashs // 2)
                in_quote = not in_quote
            else:
                i += backslashs + 1
                next_str += "\\" * (backslashs // 2) + '"'
        elif s[i] == '"':
            in_quote = not in_quote
            i += 1
        else:
            next_str += s[i]
            i += 1
    if next_str:
        ret.append(next_str)
    if in_quote:
        raise ValueError("Unmatched quote")
    return ret


if sys.platform != "win32":
    join = join_unix
    split = split_unix
    quote = quote_unix
else:
    join = join_win
    split = split_win
    quote = quote_win
