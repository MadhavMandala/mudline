from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from massprops.io.step_parser import StepParser, StepEntity


DEFAULT_DENSITY_LBM_IN3 = 0.1


def _extract_numeric_from_measure_entity(parser: StepParser, entity: StepEntity) -> Optional[float]:
    """Try to extract a numeric value from a measure entity (e.g., LENGTH_MEASURE, MASS_MEASURE)."""
    # Many measures wrap the value: LENGTH_MEASURE(1.23) or just 1.23
    if entity.name.endswith("_MEASURE") and entity.parsed_args:
        val = entity.arg(0)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _extract_unit_scale(parser: StepParser, unit_entity: StepEntity) -> float:
    """Extract conversion factor to meters / kg from a unit entity."""
    # Complex instances like (#43) are already parsed as COMPLEX with raw args
    # For simple SI units, look for SI_UNIT with MILLI/METRE etc.
    if unit_entity.name == "COMPLEX":
        raw = unit_entity.raw_args.upper()
        if ".MILLI." in raw and ".METRE." in raw:
            return 1e-3
        if ".CENTI." in raw and ".METRE." in raw:
            return 1e-2
        if ".METRE." in raw:
            return 1.0
        if "INCH" in raw:
            return 0.0254
        if "FOOT" in raw:
            return 0.3048
    elif unit_entity.name == "SI_UNIT":
        args = unit_entity.parsed_args
        prefix = None
        unit_name = None
        for a in args:
            if isinstance(a, tuple) and a[0] == "ENUM":
                val = a[1]
                if val in ("MILLI", "CENTI", "DECI", "KILO"):
                    prefix = val
                elif val in ("METRE", "GRAM", "SECOND"):
                    unit_name = val
        if unit_name == "METRE":
            if prefix == "MILLI":
                return 1e-3
            if prefix == "CENTI":
                return 1e-2
            return 1.0
    elif unit_entity.name == "CONVERSION_BASED_UNIT":
        # Look for unit name in args
        for a in unit_entity.parsed_args:
            if isinstance(a, str):
                if "INCH" in a.upper():
                    return 0.0254
                if "FOOT" in a.upper():
                    return 0.3048
    return 1.0


def _find_density_for_product_def(parser: StepParser, product_def_id: int) -> Optional[float]:
    """Search for density (kg/m^3) associated with a product definition."""
    # Pattern 1: PROPERTY_DEFINITION -> PROPERTY_DEFINITION_REPRESENTATION -> REPRESENTATION
    # with DENSITY_MEASURE_WITH_UNIT or similar
    for pd in parser.find_by_name("PROPERTY_DEFINITION"):
        if pd.arg(0) == product_def_id:
            for pdr in parser.find_by_name("PROPERTY_DEFINITION_REPRESENTATION"):
                if pdr.arg(0) == pd.eid:
                    rep = parser.resolve(pdr.arg(1))
                    if rep and rep.parsed_args:
                        items = rep.arg(1)
                        if isinstance(items, list):
                            for item_ref in items:
                                item = parser.resolve(item_ref)
                                if item:
                                    if "DENSITY" in item.name.upper():
                                        val = _extract_numeric_from_measure_entity(parser, item)
                                        if val is not None:
                                            return val
                                    # Also check for QUALIFIED_REPRESENTATION_ITEM or MEASURE_REPRESENTATION_ITEM
                                    if item.name in ("MEASURE_REPRESENTATION_ITEM", "QUALIFIED_REPRESENTATION_ITEM"):
                                        val = _extract_numeric_from_measure_entity(parser, item)
                                        if val is not None:
                                            return val

    # Pattern 2: MATERIAL_DESIGNATION linked to product
    for md in parser.find_by_name("MATERIAL_DESIGNATION"):
        defs = md.arg(1)
        if isinstance(defs, list):
            for d in defs:
                if d == product_def_id:
                    # Material found, but no density here directly
                    pass
        elif defs == product_def_id:
            pass

    # Pattern 3: Look for any entity with DENSITY in the name that references this product
    for ent in parser.entities.values():
        if "DENSITY" in ent.name.upper():
            for arg in ent.parsed_args:
                if arg == product_def_id:
                    val = _extract_numeric_from_measure_entity(parser, ent)
                    if val is not None:
                        return val

    return None


def _find_mass_and_volume_for_product_def(parser: StepParser, product_def_id: int) -> tuple[Optional[float], Optional[float]]:
    """Search for mass (kg) and volume (m^3) associated with a product definition."""
    mass = None
    volume = None
    for pd in parser.find_by_name("PROPERTY_DEFINITION"):
        if pd.arg(0) == product_def_id:
            for pdr in parser.find_by_name("PROPERTY_DEFINITION_REPRESENTATION"):
                if pdr.arg(0) == pd.eid:
                    rep = parser.resolve(pdr.arg(1))
                    if rep and rep.parsed_args:
                        items = rep.arg(1)
                        if isinstance(items, list):
                            for item_ref in items:
                                item = parser.resolve(item_ref)
                                if item:
                                    if "MASS" in item.name.upper():
                                        val = _extract_numeric_from_measure_entity(parser, item)
                                        if val is not None:
                                            mass = val
                                    if "VOLUME" in item.name.upper():
                                        val = _extract_numeric_from_measure_entity(parser, item)
                                        if val is not None:
                                            volume = val
    return mass, volume


def extract_materials(parser: StepParser) -> dict[int, dict]:
    """Extract material/density info for all product definitions.
    
    Returns a dict mapping product_definition_id -> {"density_kg_m3": float or None,
                                                      "mass_kg": float or None,
                                                      "volume_m3": float or None,
                                                      "material_name": str or None}
    """
    results: dict[int, dict] = {}
    for pd in parser.find_by_name("PRODUCT_DEFINITION"):
        pid = pd.eid
        density = _find_density_for_product_def(parser, pid)
        mass, volume = _find_mass_and_volume_for_product_def(parser, pid)
        material_name = None
        for md in parser.find_by_name("MATERIAL_DESIGNATION"):
            defs = md.arg(1)
            match = False
            if isinstance(defs, list) and pid in defs:
                match = True
            elif defs == pid:
                match = True
            if match:
                material_name = md.arg(0)
                if isinstance(material_name, str):
                    break
        results[pid] = {
            "density_kg_m3": density,
            "mass_kg": mass,
            "volume_m3": volume,
            "material_name": material_name,
        }
    return results


def apply_materials_to_tree(root, parser: StepParser) -> None:
    """Walk the component tree and assign densities from the STEP parser."""
    from massprops.utils.units import convert_density_to_internal
    materials = extract_materials(parser)

    def walk(node):
        pid = node.step_product_def_id
        info = materials.get(pid, {}) if pid is not None else {}
        density_kg_m3 = info.get("density_kg_m3")
        mass_kg = info.get("mass_kg")
        volume_m3 = info.get("volume_m3")

        if density_kg_m3 is not None:
            # kg/m^3 -> lbm/in^3
            # 1 kg = 2.20462 lbm, 1 m = 39.3701 in
            node.density = convert_density_to_internal(density_kg_m3, "kg/m**3")
            node.step_metadata["density_source"] = "STEP"
        elif mass_kg is not None and volume_m3 is not None and volume_m3 > 0:
            density_kg_m3 = mass_kg / volume_m3
            node.density = convert_density_to_internal(density_kg_m3, "kg/m**3")
            node.step_metadata["density_source"] = "computed_from_mass_volume"
        else:
            node.density = DEFAULT_DENSITY_LBM_IN3
            node.step_metadata["density_source"] = "default"
            node.step_metadata["density_flagged"] = True

        if info.get("material_name"):
            node.step_metadata["material_name"] = info["material_name"]

        for child in node.children:
            walk(child)

    walk(root)
