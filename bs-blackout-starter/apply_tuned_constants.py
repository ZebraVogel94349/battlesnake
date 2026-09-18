from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


ASSIGNMENT_RE = re.compile(
    r"^(?P<prefix>\s*)"
    r"(?P<name>[A-Z][A-Z0-9_]*)"
    r"(?P<separator>\s*=\s*)"
    r"(?P<value>[^#\r\n]*?)"
    r"(?P<trailing>\s*(?:#.*)?)"
    r"(?P<newline>\r?\n?)$"
)


def load_best_constants(results_path: Path) -> dict[str, int | float]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    constants = data.get("best_constants")
    if not isinstance(constants, dict):
        raise ValueError(f"{results_path} has no object field named 'best_constants'")

    invalid = {
        name: value
        for name, value in constants.items()
        if not isinstance(name, str) or not isinstance(value, int | float)
    }
    if invalid:
        raise ValueError(f"best_constants contains non-numeric values: {invalid}")

    return constants


def format_value(value: int | float) -> str:
    if isinstance(value, bool):
        raise ValueError("Boolean constants are not supported")
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def source_constant_names(source: str) -> set[str]:
    names = set()
    for line in source.splitlines(keepends=True):
        match = ASSIGNMENT_RE.match(line)
        if match:
            names.add(match.group("name"))
    return names


def apply_constants(
    source: str,
    constants: dict[str, int | float],
) -> tuple[str, dict[str, tuple[str, str]], set[str], set[str]]:
    changes: dict[str, tuple[str, str]] = {}
    seen: set[str] = set()
    output_lines = []

    for line in source.splitlines(keepends=True):
        match = ASSIGNMENT_RE.match(line)

        if match and match.group("name") in constants:
            name = match.group("name")
            old_value = match.group("value").strip()
            new_value = format_value(constants[name])
            seen.add(name)
            if old_value != new_value:
                changes[name] = (old_value, new_value)
                line = (
                    f"{match.group('prefix')}"
                    f"{name}"
                    f"{match.group('separator')}"
                    f"{new_value}"
                    f"{match.group('trailing')}"
                    f"{match.group('newline')}"
                )

        output_lines.append(line)

    missing = set(constants) - seen
    unchanged = seen - set(changes)
    return "".join(output_lines), changes, missing, unchanged


def validate_python_source(source: str, path: Path) -> None:
    compile(source, str(path), "exec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply best_constants from tune_best_agent_results.json to best_agent.py."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("tune_best_agent_results.json"),
        help="JSON file created by tune_best_agent.py",
    )
    parser.add_argument(
        "--agent",
        type=Path,
        default=Path("best_agent.py"),
        help="BestAgent source file to update",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing best_agent.py",
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Skip constants from the results file that are not present in the agent source.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak copy of the agent file before modifying it.",
    )
    parser.add_argument(
        "--show-unchanged",
        action="store_true",
        help="Also print constants that already match the tuned value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    constants = load_best_constants(args.results)
    source = args.agent.read_text(encoding="utf-8")
    source_names = source_constant_names(source)
    extra_constants = set(constants) - source_names

    if extra_constants and args.ignore_missing:
        constants = {name: value for name, value in constants.items() if name in source_names}

    updated_source, changes, missing, unchanged = apply_constants(source, constants)

    if missing:
        missing_list = ", ".join(sorted(missing))
        raise SystemExit(
            f"These tuned constants were not found in {args.agent}: {missing_list}\n"
            "Use --ignore-missing to skip stale constants from an older results file."
        )

    validate_python_source(updated_source, args.agent)

    if not changes:
        print(f"No changes needed. {args.agent} already contains the tuned constants.")
        if args.show_unchanged and unchanged:
            print(f"Unchanged constants: {', '.join(sorted(unchanged))}")
        return

    print(f"{'Would update' if args.dry_run else 'Updating'} {len(changes)} constants in {args.agent}:")
    for name in sorted(changes):
        old_value, new_value = changes[name]
        print(f"  {name}: {old_value} -> {new_value}")

    if args.show_unchanged and unchanged:
        print(f"\nUnchanged constants: {', '.join(sorted(unchanged))}")

    if not args.dry_run:
        if args.backup:
            backup_path = args.agent.with_name(f"{args.agent.name}.bak")
            shutil.copy2(args.agent, backup_path)
            print(f"Backup written to {backup_path}")
        args.agent.write_text(updated_source, encoding="utf-8")


if __name__ == "__main__":
    main()
