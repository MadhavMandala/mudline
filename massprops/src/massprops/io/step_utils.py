from __future__ import annotations

import glob
import os
import re
from pathlib import Path


def detect_step_unit(file_path: Path | str) -> str:
    """Scans the raw ASCII text of a STEP file to detect its native length unit."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read(100_000)
    if re.search(r"SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\.\s*\)", content, re.IGNORECASE):
        return "millimeter"
    if re.search(r"SI_UNIT\s*\(\s*\.CENTI\.\s*,\s*\.METRE\.\s*\)", content, re.IGNORECASE):
        return "centimeter"
    if re.search(r"SI_UNIT\s*\(\s*\$\s*,\s*\.METRE\.\s*\)", content, re.IGNORECASE):
        return "meter"
    if re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'INCH'", content, re.IGNORECASE):
        return "inch"
    if re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'FOOT'", content, re.IGNORECASE):
        return "foot"
    return "millimeter"


def step_file_schema(file_path: Path | str) -> str:
    """Extract the FILE_SCHEMA value from the header."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(20_000)
    m = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def build_assembly_tree(folder_path: Path | str) -> tuple[str | None, dict, dict]:
    """Parses NAUO entities in each STEP file to build the assembly hierarchy.

    Returns (root, children, name_to_file) where:
      children     — {parent: [virtual_child, ...]}; duplicate children get an @N suffix
      name_to_file — maps every virtual name → the real filename stem for mesh lookup
    """
    children: dict[str, list[str]] = {}
    name_to_file: dict[str, str] = {}
    all_original_child_names: set[str] = set()

    for file_path in glob.glob(os.path.join(folder_path, "*.stp")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        nauo_names = re.findall(
            r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\(\s*'([^']+)'", content
        )
        if not nauo_names:
            continue

        all_products = list(dict.fromkeys(
            re.findall(r"=PRODUCT\s*\(\s*'([^']+)'", content)
        ))
        child_set = set(nauo_names)
        parent = next((p for p in all_products if p not in child_set), None)
        if parent is None:
            continue

        counts: dict[str, int] = {}
        for n in nauo_names:
            counts[n] = counts.get(n, 0) + 1

        seen: dict[str, int] = {}
        virtual_children: list[str] = []
        for n in nauo_names:
            if counts[n] > 1:
                seen[n] = seen.get(n, 0) + 1
                vname = f"{n}@{seen[n]}"
            else:
                vname = n
            virtual_children.append(vname)
            name_to_file[vname] = n

        name_to_file[parent] = parent
        children[parent] = virtual_children
        all_original_child_names.update(nauo_names)

    roots = set(children) - all_original_child_names
    root = roots.pop() if roots else next(iter(children), None)
    return root, children, name_to_file


def _vname_to_display(vname: str) -> str:
    """'X1-D551101-01_-@2' → 'X1-D551101-01_2',  'X1-D551071-01_-' → 'X1-D551071-01'"""
    if "@" in vname:
        base, idx = vname.rsplit("@", 1)
        return re.sub(r"_.*$", "", base) + f"_{idx}"
    return re.sub(r"_.*$", "", vname)


def _build_parent_map(node: str, children_dict: dict, parent: str | None = None, result: dict | None = None) -> dict:
    """Returns {vname → parent_vname} for every node in the tree."""
    if result is None:
        result = {}
    result[node] = parent
    for child in children_dict.get(node, []):
        _build_parent_map(child, children_dict, node, result)
    return result


def _dfs_order(node: str, children: dict, depth: int = 0) -> list[tuple[str, int]]:
    result = [(node, depth)]
    for child in children.get(node, []):
        result.extend(_dfs_order(child, children, depth + 1))
    return result
