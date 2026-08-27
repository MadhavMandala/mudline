from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class StepEntity:
    eid: int
    name: str
    raw_args: str
    parsed_args: list[Any] = field(default_factory=list)

    def arg(self, index: int) -> Any:
        if 0 <= index < len(self.parsed_args):
            return self.parsed_args[index]
        return None


class StepParseError(Exception):
    pass


def _remove_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _parse_value(text: str, i: int) -> tuple[Any, int]:
    while i < len(text) and text[i] in " \t\n\r,":
        i += 1
    if i >= len(text):
        return None, i
    ch = text[i]
    if ch == "$":
        return None, i + 1
    if ch == "#":
        i += 1
        start = i
        while i < len(text) and text[i].isdigit():
            i += 1
        return int(text[start:i]), i
    if ch == "'":
        i += 1
        chars = []
        while i < len(text) and text[i] != "'":
            if text[i] == "\\":
                i += 1
            if i < len(text):
                chars.append(text[i])
                i += 1
        i += 1  # skip closing quote
        return "".join(chars), i
    if ch == ".":
        i += 1
        start = i
        while i < len(text) and text[i] != ".":
            i += 1
        val = text[start:i]
        i += 1
        if val in ("T", "F"):
            return val == "T", i
        return ("ENUM", val), i
    if ch == "(":
        i += 1
        items = []
        while i < len(text):
            while i < len(text) and text[i] in " \t\n\r":
                i += 1
            if i >= len(text):
                break
            if text[i] == ")":
                i += 1
                break
            val, i = _parse_value(text, i)
            items.append(val)
            while i < len(text) and text[i] in " \t\n\r":
                i += 1
            if i < len(text) and text[i] == ",":
                i += 1
        return items, i
    if ch == "*":
        return ("DERIVED",), i + 1
    # Number - only if starts with digit or sign followed by digit
    if ch.isdigit() or (ch in "+-" and i + 1 < len(text) and text[i + 1].isdigit()):
        start = i
        if ch in "+-":
            i += 1
        has_dot = False
        has_exp = False
        while i < len(text):
            c = text[i]
            if c.isdigit():
                i += 1
            elif c == "." and not has_dot and not has_exp:
                # lookahead to avoid treating lone dot as number
                if i + 1 < len(text) and text[i + 1].isdigit():
                    has_dot = True
                    i += 1
                else:
                    break
            elif c in "eE" and not has_exp:
                has_exp = True
                i += 1
                if i < len(text) and text[i] in "+-":
                    i += 1
            else:
                break
        num_str = text[start:i]
        if not num_str or num_str in "+-":
            return None, i + 1
        if has_dot or has_exp:
            return float(num_str), i
        return int(num_str), i
    # Unrecognized character; skip it
    return None, i + 1


def _parse_args(text: str) -> list[Any]:
    values = []
    i = 0
    while i < len(text):
        while i < len(text) and text[i] in " \t\n\r,":
            i += 1
        if i >= len(text):
            break
        val, i = _parse_value(text, i)
        values.append(val)
    return values


def _parse_entities(data_text: str) -> dict[int, StepEntity]:
    entities: dict[int, StepEntity] = {}
    i = 0
    while i < len(data_text):
        while i < len(data_text) and data_text[i] in " \t\n\r":
            i += 1
        if i >= len(data_text):
            break
        if data_text[i] == "#":
            i += 1
            start = i
            while i < len(data_text) and data_text[i].isdigit():
                i += 1
            eid = int(data_text[start:i])
            while i < len(data_text) and data_text[i] in " \t\n\r":
                i += 1
            if data_text[i] != "=":
                raise StepParseError(f"Expected '=' after #{eid}")
            i += 1
            while i < len(data_text) and data_text[i] in " \t\n\r":
                i += 1
            # Complex entity instance: #id = ( ... );
            if data_text[i] == "(":
                depth = 1
                i += 1
                body_start = i
                while i < len(data_text) and depth > 0:
                    if data_text[i] == "(":
                        depth += 1
                    elif data_text[i] == ")":
                        depth -= 1
                    elif data_text[i] == "'":
                        i += 1
                        while i < len(data_text) and data_text[i] != "'":
                            if data_text[i] == "\\":
                                i += 1
                            i += 1
                    i += 1
                body = data_text[body_start : i - 1]
                while i < len(data_text) and data_text[i] in " \t\n\r":
                    i += 1
                if i < len(data_text) and data_text[i] == ";":
                    i += 1
                entities[eid] = StepEntity(eid=eid, name="COMPLEX", raw_args=body)
                continue
            name_start = i
            while i < len(data_text) and (data_text[i].isalnum() or data_text[i] == "_"):
                i += 1
            name = data_text[name_start:i]
            while i < len(data_text) and data_text[i] in " \t\n\r":
                i += 1
            if data_text[i] != "(":
                raise StepParseError(f"Expected '(' after entity name {name}")
            i += 1
            arg_start = i
            depth = 1
            while i < len(data_text) and depth > 0:
                if data_text[i] == "(":
                    depth += 1
                elif data_text[i] == ")":
                    depth -= 1
                elif data_text[i] == "'":
                    i += 1
                    while i < len(data_text) and data_text[i] != "'":
                        if data_text[i] == "\\":
                            i += 1
                        i += 1
                i += 1
            args_text = data_text[arg_start : i - 1]
            while i < len(data_text) and data_text[i] in " \t\n\r":
                i += 1
            if i < len(data_text) and data_text[i] == ";":
                i += 1
            entities[eid] = StepEntity(eid=eid, name=name, raw_args=args_text)
        else:
            i += 1
    return entities


class StepParser:
    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)
        self.entities: dict[int, StepEntity] = {}
        self.header: dict[str, Any] = {}
        self._parse()

    def _parse(self) -> None:
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        text = _remove_comments(text)
        # Header
        header_match = re.search(r"HEADER\s*;(.*?)ENDSEC\s*;", text, re.DOTALL | re.IGNORECASE)
        if header_match:
            self._parse_header(header_match.group(1))
        # Data
        data_match = re.search(r"DATA\s*;(.*?)ENDSEC\s*;", text, re.DOTALL | re.IGNORECASE)
        if not data_match:
            raise StepParseError("No DATA section found")
        self.entities = _parse_entities(data_match.group(1))
        for e in self.entities.values():
            try:
                e.parsed_args = _parse_args(e.raw_args)
            except Exception as exc:
                raise StepParseError(f"Error parsing args for #{e.eid} {e.name}: {exc}") from exc

    def _parse_header(self, header_text: str) -> None:
        for m in re.finditer(r"([A-Z_0-9]+)\s*\((.*?)\)\s*;", header_text, re.DOTALL):
            key = m.group(1)
            val = m.group(2).strip()
            self.header[key] = val

    def get(self, eid: int | None) -> Optional[StepEntity]:
        if eid is None:
            return None
        return self.entities.get(eid)

    def find_by_name(self, name: str) -> list[StepEntity]:
        return [e for e in self.entities.values() if e.name == name]

    def resolve(self, value: Any) -> Any:
        if isinstance(value, int) and value in self.entities:
            return self.entities[value]
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        return value
