"""Material and mass properties carried inside a STEP file.

A STEP file is not only geometry. AP242 (ISO 10303-242) carries semantic PMI
alongside the shapes, and among it are the things geometry can never imply:
what a part is made of, how dense that material is, and sometimes the mass the
originating CAD system computed.

That matters here because the importer used to default every imported part to
one material. On a fibreglass nose cone read back as carbon, volume matched to
0.23% and the mass was still wrong by 11% -- an error that looks like a
geometry problem and is not. If the file says what the part is made of, the
tool should believe it.

Two readers, in order of trust:

    XCAF        OpenCascade's structured read. STEPCAFControl_Reader populates
                a document with a material tool, giving name, description and
                density with its unit. This is the real answer when present.

    text scan   A fallback for files whose material sits in entities XCAF does
                not surface. Cruder, and flagged as such in the report, but a
                named material with no density is still better than silently
                inheriting someone else's.

Density units are the trap. XCAF stores a unit *name* rather than a scale, and
CAD systems write g/cm3 as often as kg/m3 -- a factor of a thousand, which is
the difference between a rocket and a lump of lead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

#: Density unit names seen in the wild, mapped to a factor into kg/m^3.
DENSITY_UNITS: dict[str, float] = {
    "kg/m3": 1.0,
    "kg/m^3": 1.0,
    "kgm-3": 1.0,
    "kg_m3": 1.0,
    "g/cm3": 1000.0,
    "g/cm^3": 1000.0,
    "gcm-3": 1000.0,
    "g/mm3": 1_000_000.0,
    "kg/mm3": 1_000_000_000.0,
    "kg/dm3": 1000.0,
    "lb/in3": 27679.9047,
    "lbm/in3": 27679.9047,
}

#: Anything outside this is not a structural material and is almost certainly a
#: unit misread. Aerogel is about 1 kg/m^3; osmium is 22,590.
PLAUSIBLE_DENSITY_KG_M3 = (0.5, 30_000.0)


@dataclass
class StepMetadata:
    """What a STEP file said about itself, beyond its shape."""

    material_name: str = ""
    density_kg_m3: float = 0.0
    #: Mass as computed by the originating CAD system, when it wrote one.
    mass_kg: float = 0.0
    source: str = ""              # "xcaf", "text scan", or ""
    notes: list[str] = field(default_factory=list)

    @property
    def has_material(self) -> bool:
        return bool(self.material_name) or self.density_kg_m3 > 0

    def text(self) -> str:
        if not self.has_material:
            return "no material carried in the file"
        parts = []
        if self.material_name:
            parts.append(self.material_name)
        if self.density_kg_m3 > 0:
            parts.append(f"{self.density_kg_m3:,.0f} kg/m³")
        if self.mass_kg > 0:
            parts.append(f"{self.mass_kg:.4f} kg declared")
        return f"{', '.join(parts)} (from {self.source})"


# ----------------------------------------------------------------------


def normalise_density(value: float, unit_name: str) -> tuple[float, str]:
    """Convert a density into kg/m^3, reporting how the unit was read.

    An unrecognised unit is *not* assumed to be SI. Guessing wrong by a factor
    of a thousand is worse than declining to use the number, so an implausible
    result is rejected and the caller falls back to a named material.
    """
    if value <= 0:
        return 0.0, "no density"

    key = re.sub(r"[\s\.]", "", (unit_name or "")).lower()
    factor = DENSITY_UNITS.get(key)

    if factor is None:
        # No unit given. Infer from magnitude: a structural density in g/cm^3
        # is single digits, in kg/m^3 it is thousands.
        if value < 30.0:
            return value * 1000.0, f"assumed g/cm³ from magnitude ({value:g})"
        factor = 1.0

    converted = value * factor
    low, high = PLAUSIBLE_DENSITY_KG_M3
    if not low <= converted <= high:
        return 0.0, (
            f"density {value:g} {unit_name or '(no unit)'} reads as "
            f"{converted:,.0f} kg/m³, which is not a real material; ignored"
        )
    return converted, unit_name or "inferred"


# ----------------------------------------------------------------------


def read_xcaf(path: Path) -> StepMetadata:
    """Read material through OpenCascade's XCAF document."""
    meta = StepMetadata()
    try:
        from OCP.IFSelect import IFSelect_ReturnStatus
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.TColStd import TColStd_HSequenceOfTransient  # noqa: F401
        from OCP.TDocStd import TDocStd_Document
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
    except ImportError as exc:      # pragma: no cover - depends on the build
        meta.notes.append(f"XCAF unavailable: {exc}")
        return meta

    try:
        application = XCAFApp_Application.GetApplication_s()
        document = TDocStd_Document(TCollection_ExtendedString("MDTV-CAF"))
        application.NewDocument(TCollection_ExtendedString("MDTV-CAF"), document)

        reader = STEPCAFControl_Reader()
        reader.SetMatMode(True)
        reader.SetNameMode(True)
        status = reader.ReadFile(str(path))
        if status != IFSelect_ReturnStatus.IFSelect_RetDone:
            meta.notes.append("XCAF could not read the file")
            return meta
        if not reader.Transfer(document):
            meta.notes.append("XCAF read the file but transferred nothing")
            return meta

        tool = XCAFDoc_DocumentTool.MaterialTool_s(document.Main())
        labels = _label_sequence(tool)
        if not labels:
            return meta

        from OCP.XCAFDoc import XCAFDoc_Material

        for label in labels:
            material = XCAFDoc_Material()
            if not label.FindAttribute(XCAFDoc_Material.GetID_s(), material):
                continue
            name = _text(material.GetName())
            density_raw = float(material.GetDensity())
            unit = _text(material.GetDensValType())

            density, note = normalise_density(density_raw, unit)
            if density <= 0 and density_raw > 0:
                meta.notes.append(note)

            if name or density > 0:
                meta.material_name = name
                meta.density_kg_m3 = density
                meta.source = "xcaf"
                break
    except Exception as exc:  # noqa: BLE001
        meta.notes.append(f"XCAF read failed: {type(exc).__name__}: {exc}")

    return meta


def _label_sequence(tool) -> list:
    from OCP.TDF import TDF_LabelSequence

    labels = TDF_LabelSequence()
    tool.GetMaterialLabels(labels)
    return [labels.Value(i) for i in range(1, labels.Length() + 1)]


def _text(value) -> str:
    """OpenCascade strings arrive as HAsciiString handles or plain text."""
    if value is None:
        return ""
    for accessor in ("ToCString", "String"):
        method = getattr(value, accessor, None)
        if method is not None:
            try:
                return str(method()).strip()
            except Exception:  # noqa: BLE001
                continue
    return str(value).strip()


# ----------------------------------------------------------------------

_MATERIAL_PATTERNS = (
    # AP242 and AP203e2 spell this several ways.
    re.compile(r"MATERIAL_DESIGNATION\s*\(\s*'([^']+)'", re.IGNORECASE),
    re.compile(r"MATERIAL_PROPERTY\s*\(\s*'([^']+)'", re.IGNORECASE),
    re.compile(
        r"PROPERTY_DEFINITION\s*\(\s*'material[^']*'\s*,\s*'([^']+)'",
        re.IGNORECASE,
    ),
    re.compile(
        r"DESCRIPTIVE_REPRESENTATION_ITEM\s*\(\s*'material'\s*,\s*'([^']+)'",
        re.IGNORECASE,
    ),
)

_DENSITY_PATTERN = re.compile(
    r"'density'\s*,\s*([0-9.eE+-]+)", re.IGNORECASE
)
_MASS_PATTERN = re.compile(
    r"'mass'\s*,\s*([0-9.eE+-]+)", re.IGNORECASE
)


def read_text_scan(path: Path) -> StepMetadata:
    """Look for material entities XCAF did not surface.

    Deliberately shallow. This reads names and numbers out of the exchange
    text; it does not parse STEP. Anything it finds is reported as coming from
    a scan so the reader knows how much to trust it.
    """
    meta = StepMetadata()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        meta.notes.append(f"could not re-read the file: {exc}")
        return meta

    for pattern in _MATERIAL_PATTERNS:
        match = pattern.search(text)
        if match:
            meta.material_name = match.group(1).strip()
            meta.source = "text scan"
            break

    density = _DENSITY_PATTERN.search(text)
    if density:
        value, note = normalise_density(float(density.group(1)), "")
        if value > 0:
            meta.density_kg_m3 = value
            meta.source = "text scan"
        else:
            meta.notes.append(note)

    mass = _MASS_PATTERN.search(text)
    if mass:
        try:
            declared = float(mass.group(1))
            if 0 < declared < 1e6:
                meta.mass_kg = declared
                meta.source = meta.source or "text scan"
        except ValueError:
            pass

    return meta


def read_step_metadata(path: str | Path) -> StepMetadata:
    """Material and mass properties from a STEP file, XCAF first."""
    path = Path(path)
    meta = read_xcaf(path)
    if meta.has_material:
        return meta

    scanned = read_text_scan(path)
    scanned.notes = meta.notes + scanned.notes
    return scanned
