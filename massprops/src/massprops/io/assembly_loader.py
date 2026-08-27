from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

from massprops.io.step_parser import StepParser, StepEntity
from massprops.model.models import Component, Assembly


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


def _matrix_from_axis2_placement_3d(parser: StepParser, entity: StepEntity) -> np.ndarray:
    """Build a 4x4 matrix from an AXIS2_PLACEMENT_3D entity."""
    loc_ent = parser.resolve(entity.arg(1))
    axis_ent = parser.resolve(entity.arg(2))
    ref_ent = parser.resolve(entity.arg(3))

    origin = np.zeros(3)
    if loc_ent and loc_ent.parsed_args:
        coords = loc_ent.arg(1)
        if isinstance(coords, list) and len(coords) >= 3:
            origin = np.array([float(coords[0]), float(coords[1]), float(coords[2])])

    z_axis = np.array([0.0, 0.0, 1.0])
    if axis_ent and axis_ent.parsed_args:
        ratios = axis_ent.arg(1)
        if isinstance(ratios, list) and len(ratios) >= 3:
            z_axis = _normalize(np.array([float(ratios[0]), float(ratios[1]), float(ratios[2])]))

    x_axis = np.array([1.0, 0.0, 0.0])
    if ref_ent and ref_ent.parsed_args:
        ratios = ref_ent.arg(1)
        if isinstance(ratios, list) and len(ratios) >= 3:
            x_axis = _normalize(np.array([float(ratios[0]), float(ratios[1]), float(ratios[2])]))

    # Gram-Schmidt to get orthonormal basis
    x_axis = _normalize(x_axis - np.dot(x_axis, z_axis) * z_axis)
    y_axis = np.cross(z_axis, x_axis)

    m = np.eye(4)
    m[0:3, 0] = x_axis
    m[0:3, 1] = y_axis
    m[0:3, 2] = z_axis
    m[0:3, 3] = origin
    return m


def _matrix_from_cartesian_transformation_operator_3d(parser: StepParser, entity: StepEntity) -> np.ndarray:
    """Build a 4x4 matrix from a CARTESIAN_TRANSFORMATION_OPERATOR_3D entity."""
    axis1_ent = parser.resolve(entity.arg(1))
    axis2_ent = parser.resolve(entity.arg(2))
    origin_ent = parser.resolve(entity.arg(3))
    scale = 1.0
    if len(entity.parsed_args) > 4 and entity.arg(4) is not None:
        scale = float(entity.arg(4))

    x_axis = np.array([1.0, 0.0, 0.0])
    if axis1_ent and axis1_ent.parsed_args:
        ratios = axis1_ent.arg(1)
        if isinstance(ratios, list) and len(ratios) >= 3:
            x_axis = _normalize(np.array([float(ratios[0]), float(ratios[1]), float(ratios[2])]))

    y_axis = np.array([0.0, 1.0, 0.0])
    if axis2_ent and axis2_ent.parsed_args:
        ratios = axis2_ent.arg(1)
        if isinstance(ratios, list) and len(ratios) >= 3:
            y_axis = _normalize(np.array([float(ratios[0]), float(ratios[1]), float(ratios[2])]))

    # Orthonormalize
    y_axis = _normalize(y_axis - np.dot(y_axis, x_axis) * x_axis)
    z_axis = np.cross(x_axis, y_axis)

    origin = np.zeros(3)
    if origin_ent and origin_ent.parsed_args:
        coords = origin_ent.arg(1)
        if isinstance(coords, list) and len(coords) >= 3:
            origin = np.array([float(coords[0]), float(coords[1]), float(coords[2])])

    m = np.eye(4)
    m[0:3, 0] = scale * x_axis
    m[0:3, 1] = scale * y_axis
    m[0:3, 2] = scale * z_axis
    m[0:3, 3] = origin
    return m


def _extract_transform_from_item_defined_transformation(parser: StepParser, entity: StepEntity) -> np.ndarray:
    """Extract transform from ITEM_DEFINED_TRANSFORMATION."""
    item1 = parser.resolve(entity.arg(1))
    item2 = parser.resolve(entity.arg(2))
    # In many STEP files, transform_item_2 is the placement in parent space
    if item2 and item2.name == "AXIS2_PLACEMENT_3D":
        return _matrix_from_axis2_placement_3d(parser, item2)
    return np.eye(4)


def _find_transform_for_nauo(parser: StepParser, nauo_id: int) -> np.ndarray:
    """Find the transform matrix associated with a NAUO entity."""
    for cdsr in parser.find_by_name("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"):
        relation = parser.resolve(cdsr.arg(1))
        if relation and relation.eid == nauo_id:
            rep_rel = parser.resolve(cdsr.arg(0))
            if not rep_rel:
                continue
            # rep_rel could be SHAPE_REPRESENTATION_RELATIONSHIP or WITH_TRANSFORMATION variant
            transform_op = None
            if rep_rel.name in ("REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION", "SHAPE_REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION"):
                transform_op = parser.resolve(rep_rel.arg(3))
            elif rep_rel.name == "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION_AND_SHAPE_REPRESENTATION":
                transform_op = parser.resolve(rep_rel.arg(3))
            if transform_op:
                if transform_op.name == "ITEM_DEFINED_TRANSFORMATION":
                    return _extract_transform_from_item_defined_transformation(parser, transform_op)
                elif transform_op.name == "CARTESIAN_TRANSFORMATION_OPERATOR_3D":
                    return _matrix_from_cartesian_transformation_operator_3d(parser, transform_op)
    return np.eye(4)


def _find_external_file(parser: StepParser, product_def_id: int) -> Optional[Path]:
    """Look for external file references associated with a product definition."""
    # Pattern 1: PRODUCT_DEFINITION_WITH_ASSOCIATED_DOCUMENTS -> DOCUMENT -> DOCUMENT_FILE
    for pdwad in parser.find_by_name("PRODUCT_DEFINITION_WITH_ASSOCIATED_DOCUMENTS"):
        if pdwad.arg(0) == product_def_id:
            docs = pdwad.arg(1)
            if isinstance(docs, list):
                for doc_ref in docs:
                    doc = parser.resolve(doc_ref)
                    if doc:
                        for df in parser.find_by_name("DOCUMENT_FILE"):
                            if df.arg(0) == doc.eid:
                                fname = df.arg(2)
                                if isinstance(fname, str):
                                    return Path(fname)
            elif docs:
                doc = parser.resolve(docs)
                if doc:
                    for df in parser.find_by_name("DOCUMENT_FILE"):
                        if df.arg(0) == doc.eid:
                            fname = df.arg(2)
                            if isinstance(fname, str):
                                return Path(fname)

    # Pattern 2: IDENTIFICATION_ASSIGNMENT on EXTERNALLY_DEFINED_ITEM related to shape
    for edi in parser.find_by_name("EXTERNALLY_DEFINED_ITEM"):
        for ida in parser.find_by_name("IDENTIFICATION_ASSIGNMENT"):
            if ida.arg(0) == edi.eid:
                role = parser.resolve(ida.arg(2))
                if role and role.name == "IDENTIFICATION_ROLE":
                    role_name = role.arg(0)
                    if isinstance(role_name, str) and "external" in role_name.lower():
                        fname = ida.arg(1)
                        if isinstance(fname, str):
                            return Path(fname)

    # Pattern 3: APPLIED_DOCUMENT_REFERENCE -> DOCUMENT_FILE
    for adr in parser.find_by_name("APPLIED_DOCUMENT_REFERENCE"):
        products = adr.arg(2)
        if isinstance(products, list):
            product_ids = products
        elif products is not None:
            product_ids = [products]
        else:
            continue
        if product_def_id not in product_ids:
            continue
        doc = parser.resolve(adr.arg(0))
        if doc and doc.name == "DOCUMENT_FILE":
            # Filename is typically the first argument; fall back to second
            for idx in (0, 1):
                fname = doc.arg(idx)
                if isinstance(fname, str) and fname.endswith(".stp"):
                    return Path(fname)
            # Also try arg(0) or arg(1) without the .stp check as last resort
            for idx in (0, 1):
                fname = doc.arg(idx)
                if isinstance(fname, str):
                    return Path(fname)

    return None


def _build_component(parser: StepParser, product_def_id: int, visited: Optional[set[int]] = None) -> Component:
    """Recursively build a Component/Assembly tree from a product definition."""
    if visited is None:
        visited = set()
    if product_def_id in visited:
        # Prevent cycles
        return Component(name="<cycle>")
    visited.add(product_def_id)

    product_def = parser.get(product_def_id)
    name = ""
    if product_def:
        formation = parser.resolve(product_def.arg(2))
        if formation:
            product = parser.resolve(formation.arg(2))
            if product:
                name = product.arg(1) or ""

    # Determine if this product is a parent in any NAUO
    is_assembly = False
    children_nauos = []
    for nauo in parser.find_by_name("NEXT_ASSEMBLY_USAGE_OCCURRENCE"):
        parent_id = nauo.arg(3)
        if parent_id == product_def_id:
            is_assembly = True
            children_nauos.append(nauo)

    if is_assembly:
        comp = Assembly(name=name)
    else:
        comp = Component(name=name)

    comp.step_product_def_id = product_def_id
    comp.instance_transform = _find_transform_for_nauo(parser, product_def_id)
    # Note: transform for root is typically identity; for children it's found via their NAUO

    ext_file = _find_external_file(parser, product_def_id)
    if ext_file:
        comp.source_step = ext_file

    # For assemblies, build children
    for nauo in children_nauos:
        child_id = nauo.arg(4)
        child_comp = _build_component(parser, child_id, visited.copy())
        child_comp.instance_transform = _find_transform_for_nauo(parser, nauo.eid)
        comp.children.append(child_comp)

    return comp


def load_from_folder(folder_path: Path | str) -> tuple[Assembly, Path]:
    """Find the master assembly file in a folder.

    Picks the file with the most children that is not referenced as a child
    by any other assembly in the folder (the 'true root').

    Returns (root, master_path).
    """
    import glob
    folder_path = Path(folder_path)
    stp_files = sorted(
        glob.glob(str(folder_path / "*.stp")) + glob.glob(str(folder_path / "*.step"))
    )
    if not stp_files:
        raise ValueError(f"No .stp/.step files found in {folder_path}")

    candidates = []
    for candidate in stp_files:
        try:
            root = load_assembly(candidate)
            candidates.append((root, Path(candidate)))
        except Exception:
            continue

    if not candidates:
        raise ValueError(f"No valid STEP files found in {folder_path}")

    # Build set of filenames that are referenced as external children
    referenced_names = set()
    for root, _ in candidates:
        for child in root.children:
            if child.source_step:
                referenced_names.add(Path(child.source_step).name)

    # Prefer candidates that are NOT referenced by others (true roots)
    top_level = [(r, p) for r, p in candidates if p.name not in referenced_names]
    pool = top_level if top_level else candidates

    # From the pool, pick the one with the most children
    best_root, best_path = max(pool, key=lambda item: len(item[0].children))
    return best_root, best_path


def expand_external_references(
    node: Component,
    folder_path: Path | str,
    visited: Optional[set[str]] = None,
) -> None:
    """Recursively replace children whose source_step points to another STEP file
    in the same folder with the loaded assembly tree from that file.
    """
    folder_path = Path(folder_path)
    if visited is None:
        visited = set()

    for i, child in enumerate(node.children):
        if child.source_step:
            source = Path(child.source_step)
            # Resolve relative to the folder if needed
            if not source.is_absolute():
                source = folder_path / source.name
            if source.exists() and source.suffix.lower() in (".stp", ".step"):
                key = str(source.resolve())
                if key not in visited:
                    visited.add(key)
                    try:
                        loaded = load_assembly(source)
                        # Flatten single-part wrappers: if loaded is an Assembly
                        # with exactly 1 child of the same name, use the child.
                        if (
                            isinstance(loaded, Assembly)
                            and len(loaded.children) == 1
                            and loaded.children[0].name == loaded.name
                        ):
                            loaded = loaded.children[0]
                        # Preserve the transform from the parent assembly
                        loaded.instance_transform = child.instance_transform
                        loaded.source_step = source
                        node.children[i] = loaded
                        # Recurse into the newly loaded assembly
                        expand_external_references(loaded, folder_path, visited)
                    except Exception:
                        pass
        # Recurse into children that were not replaced
        expand_external_references(node.children[i], folder_path, visited)


def resolve_source_paths(node: Component, folder_path: Path | str) -> None:
    """Ensure every source_step in the tree is an absolute Path."""
    folder_path = Path(folder_path)
    if node.source_step is not None:
        sp = Path(node.source_step)
        if not sp.is_absolute():
            node.source_step = folder_path / sp.name
    for child in node.children:
        resolve_source_paths(child, folder_path)


def load_assembly(file_path: Path | str) -> Assembly:
    """Load an AP242 STEP assembly file and return the root Assembly."""
    file_path = Path(file_path)
    parser = StepParser(file_path)

    # Find root product definitions (those that are never children)
    all_parents = set()
    all_children = set()
    for nauo in parser.find_by_name("NEXT_ASSEMBLY_USAGE_OCCURRENCE"):
        all_parents.add(nauo.arg(3))
        all_children.add(nauo.arg(4))

    root_ids = all_parents - all_children
    if not root_ids:
        # No assembly structure; treat all product definitions as roots
        root_ids = {pd.eid for pd in parser.find_by_name("PRODUCT_DEFINITION")}

    if not root_ids:
        raise ValueError("No product definitions found in STEP file")

    # If multiple roots, create a virtual root
    if len(root_ids) == 1:
        root_id = root_ids.pop()
        root = _build_component(parser, root_id)
        if not isinstance(root, Assembly):
            # Single part file
            root = Assembly(name=root.name)
            root.children.append(_build_component(parser, root_id))
        root.source_step = file_path
        return root
    else:
        root = Assembly(name=file_path.stem)
        root.source_step = file_path
        for rid in root_ids:
            child = _build_component(parser, rid)
            root.children.append(child)
        return root
