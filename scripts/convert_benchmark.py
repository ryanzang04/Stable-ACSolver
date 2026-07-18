#!/usr/bin/env python3
"""Convert a two-generator AC benchmark to the ``1190MS.txt`` format.

The input may be Avi's benchmark CSV/JSON (metadata is ignored except for
``r1`` and ``r2``), or contain one presentation per line in any of these forms::

    <x, y | xyX, yxY>
    xyX, yxY
    ["xyX", "yxY"]
    [[1, 2, -1], [2, 1, -2]]
    [1, 2, -1, 0, 0, 2, 1, -2, 0, 0]  # already padded

Compact words use lower-case letters for generators and upper-case letters for
their inverses.  Algebraic exponents are also accepted (for example,
``x*y^-2*X``).  Blank lines, comment lines, and an ``r1,r2`` CSV header are
ignored.

The output has one Python-literal flat list per line.  Each relator is freely
and cyclically reduced, canonicalized over cyclic rotations and inversion, the
two relators are sorted, and each is zero-padded to ``--max-length``.  These are
the same representation and canonicalization conventions as ``data/1190MS.txt``.

Example, from the repository root::

    python scripts/convert_benchmark.py combined_benchmark.csv data/Avi66.txt

Use ``--generators ab`` if Avi's file calls the two generators ``a`` and ``b``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


class ParseError(ValueError):
    """An input row could not be interpreted as one presentation."""


_SUPERSCRIPTS = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
    }
)
_SUPERSCRIPT_CHARS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"


def free_reduce(word: Sequence[int]) -> list[int]:
    """Freely reduce an integer-encoded group word."""
    reduced: list[int] = []
    for value in word:
        if value == 0:
            continue
        if reduced and reduced[-1] == -value:
            reduced.pop()
        else:
            reduced.append(value)
    return reduced


def cyclic_reduce(word: Sequence[int]) -> list[int]:
    """Freely reduce a word and cancel inverse letters at its two ends."""
    reduced = free_reduce(word)
    start = 0
    end = len(reduced)
    while end - start >= 2 and reduced[start] == -reduced[end - 1]:
        start += 1
        end -= 1
    return reduced[start:end]


def inverse_word(word: Sequence[int]) -> tuple[int, ...]:
    return tuple(-value for value in reversed(word))


def letter_order(value: int, number_of_generators: int = 2) -> int:
    """Return the notebook order, generalized from ``Y < y < X < x``."""
    return 2 * (number_of_generators - abs(value)) + (1 if value > 0 else 0)


def word_order_key(
    word: Sequence[int], number_of_generators: int = 2
) -> tuple[int, ...]:
    return tuple(letter_order(value, number_of_generators) for value in word)


def canonical_word(
    word: Sequence[int], number_of_generators: int = 2
) -> tuple[int, ...]:
    """Return the least rotation of a cyclic word or of its inverse."""
    reduced = tuple(cyclic_reduce(word))
    if not reduced:
        return ()
    inverse = inverse_word(reduced)
    candidates = []
    for candidate in (reduced, inverse):
        candidates.extend(
            candidate[offset:] + candidate[:offset]
            for offset in range(len(candidate))
        )
    return min(candidates, key=lambda item: word_order_key(item, number_of_generators))


def canonical_presentation(
    relators: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(relators) != 2:
        raise ParseError(f"expected exactly two relators, got {len(relators)}")
    canonical = [canonical_word(word) for word in relators]
    if any(not word for word in canonical):
        raise ParseError("empty relators are not valid benchmark entries")
    # 1190MS orders the two fixed-width integer blocks.  Appending their first
    # padding zero gives the same comparison without needing max_length here.
    canonical.sort(key=lambda word: word + (0,))
    return canonical[0], canonical[1]


def _expand_superscripts(text: str) -> str:
    """Turn x⁻² into x^-2 without changing ordinary text."""
    output: list[str] = []
    index = 0
    while index < len(text):
        if text[index] not in _SUPERSCRIPT_CHARS:
            output.append(text[index])
            index += 1
            continue
        end = index
        while end < len(text) and text[end] in _SUPERSCRIPT_CHARS:
            end += 1
        output.append("^" + text[index:end].translate(_SUPERSCRIPTS))
        index = end
    return "".join(output)


def parse_word(value: Any, generators: str = "xy") -> list[int]:
    """Parse a compact/algebraic word or an integer sequence."""
    if isinstance(value, (list, tuple)):
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            raise ParseError(f"word is not a list of integers: {value!r}")
        invalid = [item for item in value if abs(item) > len(generators)]
        if invalid:
            raise ParseError(
                f"generator code {invalid[0]} is outside 1..{len(generators)}"
            )
        return [item for item in value if item]
    if not isinstance(value, str):
        raise ParseError(f"word must be text or an integer list, got {value!r}")

    text = value.strip().strip('"\'')
    if text in {"", "1", "IdWord", "id"}:
        return []
    text = _expand_superscripts(text)
    text = re.sub(r"\^\{\s*([+-]?\d+)\s*\}", r"^\1", text)

    codes = {letter: index for index, letter in enumerate(generators, start=1)}
    codes.update({letter.upper(): -index for index, letter in enumerate(generators, start=1)})
    result: list[int] = []
    position = 0
    token = re.compile(r"([A-Za-z])(?:\^\(?([+-]?\d+)\)?)?")
    while position < len(text):
        if text[position].isspace() or text[position] in "*.":
            position += 1
            continue
        match = token.match(text, position)
        if not match:
            raise ParseError(
                f"cannot parse word {value!r} at {text[position:]!r}"
            )
        letter, exponent_text = match.groups()
        if letter not in codes:
            raise ParseError(
                f"unknown generator {letter!r}; expected {generators!r} "
                "(upper case means inverse)"
            )
        exponent = int(exponent_text) if exponent_text is not None else 1
        code = codes[letter]
        if exponent < 0:
            code = -code
        result.extend([code] * abs(exponent))
        position = match.end()
    return result


def _is_int_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )


def _is_word_object(value: Any) -> bool:
    return isinstance(value, str) or _is_int_sequence(value)


def _split_flat_presentation(values: Sequence[int]) -> list[list[int]]:
    if len(values) < 2 or len(values) % 2:
        raise ParseError(
            "a flat presentation must have an even number of integer entries"
        )
    midpoint = len(values) // 2
    blocks = [list(values[:midpoint]), list(values[midpoint:])]
    for block_number, block in enumerate(blocks, start=1):
        seen_padding = False
        for value in block:
            if value == 0:
                seen_padding = True
            elif seen_padding:
                raise ParseError(
                    f"nonzero entry after padding in relator {block_number}"
                )
    return [[value for value in block if value] for block in blocks]


def parse_presentation_object(value: Any, generators: str = "xy") -> list[list[int]]:
    """Parse a structured Python/JSON object containing one presentation."""
    if isinstance(value, dict):
        if "relators" in value:
            value = value["relators"]
        elif "presentation" in value:
            value = value["presentation"]
        elif "r1" in value and "r2" in value:
            value = [value["r1"], value["r2"]]
        else:
            raise ParseError("object needs relators, presentation, or r1/r2 keys")

    if _is_int_sequence(value):
        return _split_flat_presentation(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ParseError("expected a pair of relators")
    if not all(_is_word_object(word) for word in value):
        raise ParseError("expected a pair of word strings or integer lists")
    return [parse_word(word, generators) for word in value]


def _strip_presentation_wrapper(line: str) -> str:
    # GAP/mathematical notation: <x, y | r1, r2> or <r1, r2>.
    if line.startswith("<") and line.endswith(">"):
        line = line[1:-1].strip()
        if "|" in line:
            line = line.rsplit("|", 1)[1].strip()
    return line


def _csv_fields(line: str, delimiter: str) -> list[str]:
    return [field.strip() for field in next(csv.reader([line], delimiter=delimiter))]


def parse_presentation_line(line: str, generators: str = "xy") -> list[list[int]]:
    """Parse one nonblank, noncomment input line."""
    line = _strip_presentation_wrapper(line.strip())
    try:
        literal = ast.literal_eval(line)
    except (SyntaxError, ValueError):
        literal = None
    else:
        return parse_presentation_object(literal, generators)

    fields: list[str] | None = None
    for delimiter in ("\t", ",", ";", "|"):
        if delimiter in line:
            candidate = _csv_fields(line, delimiter)
            if len(candidate) >= 2:
                fields = candidate
                break
    if fields is None:
        fields = line.split()

    # Permit an identifier/index column before the two relators.
    if len(fields) == 3:
        fields = fields[1:]
    if len(fields) != 2:
        raise ParseError(
            "expected two relators separated by a comma, semicolon, tab, or pipe"
        )
    return [parse_word(field, generators) for field in fields]


def _is_header(line: str) -> bool:
    normalized = re.sub(r"[\s_\-\"']", "", line.lower())
    return normalized in {
        "r1,r2",
        "relator1,relator2",
        "index,r1,r2",
        "id,r1,r2",
        "name,r1,r2",
    }


def iter_presentations(
    lines: Iterable[str], generators: str = "xy"
) -> Iterable[tuple[int, list[list[int]]]]:
    """Yield ``(line_number, relators)`` while retaining useful diagnostics."""
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith(("#", "//")) or _is_header(line):
            continue
        # Inline comments are useful in hand-authored word-pair files.  Do not
        # mistake a leading comment for data after stripping it.
        line = re.sub(r"\s+#.*$", "", line).strip()
        if not line:
            continue
        try:
            yield line_number, parse_presentation_line(line, generators)
        except ParseError as error:
            raise ParseError(f"line {line_number}: {error}") from error


def _records_from_json(value: Any) -> list[Any]:
    """Extract presentation records from Avi's JSON container shapes."""
    if isinstance(value, dict):
        if "r1" in value and "r2" in value:
            return [value]
        for key in ("rows", "subset", "presentations", "data"):
            if key in value:
                records = value[key]
                if not isinstance(records, list):
                    raise ParseError(f"JSON field {key!r} must be a list")
                return records
        raise ParseError(
            "JSON object needs r1/r2 or a rows, subset, presentations, or data list"
        )
    if isinstance(value, list):
        # A flat integer list or pair of relators is one presentation; all
        # other lists are interpreted as a collection of presentations.
        try:
            parse_presentation_object(value)
        except ParseError:
            return value
        return [value]
    raise ParseError("top-level JSON value must be an object or list")


def read_presentations(
    input_path: Path, generators: str = "xy"
) -> list[tuple[str, list[list[int]]]]:
    """Read Avi CSV/JSON or the older one-presentation-per-line format."""
    text = input_path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ParseError("input contains no presentations")

    # Avi's JSON files are top-level metadata objects containing either a
    # ``rows`` (combined benchmark) or ``subset`` (ladder benchmark) list.
    if text.lstrip().startswith(("{", "[")):
        try:
            json_value = json.loads(text)
        except json.JSONDecodeError:
            # The 1190MS-style line format also begins with '[' but is not one
            # JSON document, so fall through to the line parser.
            pass
        else:
            parsed: list[tuple[str, list[list[int]]]] = []
            for index, record in enumerate(_records_from_json(json_value), start=1):
                try:
                    relators = parse_presentation_object(record, generators)
                except ParseError as error:
                    raise ParseError(f"JSON record {index}: {error}") from error
                parsed.append((f"JSON record {index}", relators))
            return parsed

    # Avi's CSV includes many experiment columns; r1 and r2 are the complete
    # presentation and every other column is metadata that must not enter the
    # fixed-width model input.
    csv_reader = csv.DictReader(io.StringIO(text))
    if csv_reader.fieldnames and {"r1", "r2"}.issubset(csv_reader.fieldnames):
        parsed = []
        for csv_row_number, record in enumerate(csv_reader, start=2):
            try:
                relators = parse_presentation_object(record, generators)
            except ParseError as error:
                raise ParseError(f"CSV row {csv_row_number}: {error}") from error
            parsed.append((f"CSV row {csv_row_number}", relators))
        if not parsed:
            raise ParseError("CSV contains a header but no presentations")
        return parsed

    return [
        (f"line {line_number}", relators)
        for line_number, relators in iter_presentations(text.splitlines(), generators)
    ]


def encode_presentation(
    relators: Sequence[Sequence[int]], max_length: int
) -> list[int]:
    canonical = canonical_presentation(relators)
    longest = max(map(len, canonical))
    if longest > max_length:
        raise ParseError(
            f"canonical relator length {longest} exceeds --max-length {max_length}"
        )
    flat: list[int] = []
    for word in canonical:
        flat.extend(word)
        flat.extend([0] * (max_length - len(word)))
    return flat


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    generators: str = "xy",
    max_length: int = 24,
    deduplicate: bool = False,
) -> tuple[int, int]:
    """Convert a file and return ``(rows_written, duplicates_removed)``."""
    if input_path.resolve() == output_path.resolve():
        raise ParseError("input and output paths must be different")
    seen: set[tuple[int, ...]] = set()
    rows: list[list[int]] = []
    duplicates = 0
    for location, relators in read_presentations(input_path, generators):
        try:
            encoded = encode_presentation(relators, max_length)
        except ParseError as error:
            raise ParseError(f"{location}: {error}") from error
        key = tuple(encoded)
        if deduplicate and key in seen:
            duplicates += 1
            continue
        seen.add(key)
        rows.append(encoded)
    if not rows:
        raise ParseError("input contains no presentations")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            for row in rows:
                destination.write(repr(row) + "\n")
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(rows), duplicates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Avi's source benchmark")
    parser.add_argument("output", type=Path, help="destination .txt dataset")
    parser.add_argument(
        "--generators",
        default="xy",
        help="two lower-case input generator letters (default: xy)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=24,
        help="entries reserved for each relator (default: 24, as in 1190MS)",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="remove duplicate canonical presentations (default: preserve rows)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.generators) != 2 or not args.generators.isalpha() or not args.generators.islower():
        parser.error("--generators must be two distinct lower-case letters")
    if len(set(args.generators)) != 2:
        parser.error("--generators must contain two distinct letters")
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    try:
        rows, duplicates = convert_file(
            args.input,
            args.output,
            generators=args.generators,
            max_length=args.max_length,
            deduplicate=args.deduplicate,
        )
    except (OSError, ParseError) as error:
        parser.exit(1, f"error: {error}\n")
    message = f"wrote {rows} presentations to {args.output}"
    if args.deduplicate:
        message += f" ({duplicates} duplicates removed)"
    print(message, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
