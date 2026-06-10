#!/usr/bin/env python3
"""Post-process Pandoc memoir .tex before Tectonic (inline math + table macro)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

MATH_BLOCK = re.compile(r"(\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|\$[^$\n]+\$)", re.DOTALL)


def fix_plain(text: str) -> str:
    text = text.replace("(c \\approx \\sqrt{e})", "$c \\approx \\sqrt{e}$")
    text = re.sub(r"(?<!\\)§", r"\\S{}", text)
    text = re.sub(r"\(n \\textgreater\{\} 2\)", r"$n > 2$", text)
    text = re.sub(r"\(c, n, K\)", r"$(c, n, K)$", text)
    text = re.sub(r"\(\\cos\\phi\)", r"$\\cos\\phi$", text)
    text = re.sub(r"\(\\sin\\phi\)", r"$\\sin\\phi$", text)
    text = re.sub(
        r"\(c \\approx \\sqrt\{e\} \\approx 1\.65\)",
        r"$c \\approx \\sqrt{e} \\approx 1.65$",
        text,
    )
    text = re.sub(r"\(c \\approx 1\.65\)", r"$c \\approx 1.65$", text)
    text = re.sub(r"\(c \\approx \\sqrt\{e\}\)", r"$c \\approx \\sqrt{e}$", text)
    text = re.sub(
        r"scale factor \(c\s*\n\\approx \\sqrt\{e\} \\approx 1\.65\)",
        r"scale factor $c \\approx \\sqrt{e} \\approx 1.65$",
        text,
    )
    text = re.sub(r"\(([cn])\)", r"$\1$", text)
    text = re.sub(r"\(\\mathbf\{([^}]+)\}\)", r"$\\mathbf{\1}$", text)
    text = re.sub(
        r"\(\\mathbf\{([a-zA-Z])\}\\?_\{?([a-zA-Z0-9\\]+)\}?\)",
        r"$\\mathbf{\1}_{\2}$",
        text,
    )
    text = re.sub(
        r"\(\\(alpha|beta|gamma|delta|epsilon|lambda|mu|sigma|theta|phi|psi|omega)\)",
        r"$\\\1$",
        text,
    )
    return text


NARROW_SCRIPT_TABLE = (
    "  >{\\raggedright\\arraybackslash}p{(\\linewidth - 4\\tabcolsep) * \\real{0.1633}}\n"
    "  >{\\raggedright\\arraybackslash}p{(\\linewidth - 4\\tabcolsep) * \\real{0.5102}}\n"
    "  >{\\raggedright\\arraybackslash}p{(\\linewidth - 4\\tabcolsep) * \\real{0.3265}}@{}"
)
WIDE_SCRIPT_TABLE = (
    "  >{\\raggedright\\arraybackslash}p{(\\linewidth - 4\\tabcolsep) * \\real{0.32}}\n"
    "  >{\\raggedright\\arraybackslash}p{(\\linewidth - 4\\tabcolsep) * \\real{0.36}}\n"
    "  >{\\raggedright\\arraybackslash}p{(\\linewidth - 4\\tabcolsep) * \\real{0.32}}@{}"
)


def patch(text: str) -> str:
    if "\\newcommand{\\real}" not in text and "\\real{" in text:
        text = text.replace(
            "\\usepackage{calc}",
            "\\usepackage{calc}\n\\newcommand{\\real}[1]{#1}",
            1,
        )
    text = text.replace(NARROW_SCRIPT_TABLE, WIDE_SCRIPT_TABLE)
    text = re.sub(
        r"\\texttt\{(results/[^}]+)\}",
        r"{\\ttfamily\\nolinkurl{\1}}",
        text,
    )
    text = re.sub(
        r"\\texttt\{(run\\_[^}]+)\}",
        lambda m: "{\\ttfamily " + m.group(1).replace("\\_", "\\_\\allowbreak ") + "}",
        text,
    )
    parts: list[str] = []
    last = 0
    for match in MATH_BLOCK.finditer(text):
        parts.append(fix_plain(text[last : match.start()]))
        parts.append(match.group(0))
        last = match.end()
    parts.append(fix_plain(text[last:]))
    return "".join(parts)


def main() -> None:
    path = Path(sys.argv[1])
    path.write_text(patch(path.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
