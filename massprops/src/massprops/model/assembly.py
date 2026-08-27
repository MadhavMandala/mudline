from __future__ import annotations

import numpy as np
from typing import Optional

from massprops.model.models import Component, Assembly, MassProperties


def _leaf_descendants(node: Component) -> list[Component]:
    """Return all leaf nodes under a given node."""
    if not node.children:
        return [node]
    leaves = []
    for child in node.children:
        leaves.extend(_leaf_descendants(child))
    return leaves


def _scalable_leaves(node: Component) -> list[Component]:
    """Return leaf descendants, stopping at sub-assemblies with mass overrides.

    Leaves inside a sub-assembly that has its own mass override are excluded,
    because that sub-assembly manages its own scaling.
    """
    if not node.children:
        return [node]
    leaves = []
    for child in node.children:
        if 'mass' in child.override_fields and child.overridden_props is not None:
            continue
        leaves.extend(_scalable_leaves(child))
    return leaves


def _raw_leaf_mass(leaf: Component) -> float:
    """Return a leaf's mass without applying any mass_scale factor."""
    base = leaf.computed_props or MassProperties()
    if leaf.overridden_props is not None and leaf.override_fields:
        return leaf.overridden_props.mass if 'mass' in leaf.override_fields else base.mass
    return base.mass


def _reset_all_mass_scales(node: Component) -> None:
    """Reset mass_scale to 1.0 on node and all descendants."""
    node.mass_scale = 1.0
    for child in node.children:
        _reset_all_mass_scales(child)


def _apply_assembly_mass_override(node: Component, target_mass: float) -> None:
    """Set mass_scale on leaf descendants so the assembly total hits target_mass.

    Direct children with mass overrides and leaves with mass overrides are treated
    as FIXED (mass_scale = 1.0). All other scalable leaves are scaled uniformly
    so the assembly total matches the target.

    Uses raw leaf masses (ignores any existing mass_scale) so recomputation is
    idempotent.
    """
    leaves = _scalable_leaves(node)

    fixed_mass = 0.0
    flexible_mass = 0.0

    # Direct children with overrides are fixed black-boxes
    for child in node.children:
        if 'mass' in child.override_fields and child.overridden_props is not None:
            fixed_mass += aggregate_properties(child).mass

    # Scalable leaves are either fixed (leaf override) or flexible
    for leaf in leaves:
        raw = _raw_leaf_mass(leaf)
        if 'mass' in leaf.override_fields and leaf.overridden_props is not None:
            fixed_mass += raw
            leaf.mass_scale = 1.0
        else:
            flexible_mass += raw

    remaining = target_mass - fixed_mass
    if remaining < -1e-9:
        raise ValueError(
            f"Fixed component masses ({fixed_mass:.4f}) exceed assembly target ({target_mass:.4f})"
        )

    if flexible_mass > 1e-12:
        flexible_scale = remaining / flexible_mass
    elif remaining > 1e-12:
        raise ValueError(
            f"No un-overridden components to absorb remaining mass ({remaining:.4f})"
        )
    else:
        flexible_scale = 1.0  # Nothing to scale

    for leaf in leaves:
        if not ('mass' in leaf.override_fields and leaf.overridden_props is not None):
            leaf.mass_scale = flexible_scale


def aggregate_properties(node: Component, apply_mass_scale: bool = True) -> MassProperties:
    """Recursively compute aggregated mass properties for an Assembly or Component.

    Uses the parallel axis theorem to shift child inertias to the parent frame.
    Respects overrides: if a leaf has overridden_props, those are used instead of computed_props.

    For assemblies with a mass override, this dynamically recomputes the required
    mass_scale for leaf descendants based on the current raw total mass, ensuring
    the assembly total always matches the override even when children change.
    Leaves or sub-assemblies that have their own mass override are treated as fixed
    and are not scaled.
    """
    if not node.children:
        # Leaf node
        base = node.computed_props or MassProperties()

        # Apply field-level overrides
        if node.overridden_props is not None and node.override_fields:
            mass = node.overridden_props.mass if 'mass' in node.override_fields else base.mass
            cg = node.overridden_props.cg if 'cg' in node.override_fields else base.cg
            inertia = node.overridden_props.inertia if 'inertia' in node.override_fields else base.inertia
            volume = node.overridden_props.volume if 'volume' in node.override_fields else base.volume
        else:
            mass = base.mass
            cg = base.cg
            inertia = base.inertia
            volume = base.volume

        # Apply assembly-distributed mass scale only when requested
        if apply_mass_scale and node.mass_scale != 1.0 and mass > 0:
            mass = mass * node.mass_scale
            inertia = inertia * node.mass_scale

        return MassProperties(mass=mass, cg=cg, inertia=inertia, volume=volume)

    # Assembly node
    # Determine own geometry properties
    if node.children and 'mass' in node.override_fields and node.overridden_props is not None:
        own = node.computed_props or MassProperties()
    else:
        own = node.overridden_props or node.computed_props or MassProperties()

    has_mass_override = 'mass' in node.override_fields and node.overridden_props is not None

    if has_mass_override:
        target_mass = node.overridden_props.mass
        target_for_children = target_mass - own.mass

        if target_for_children < -1e-9:
            node.step_metadata['mass_error'] = (
                f"Own geometry mass ({own.mass:.4f}) exceeds target ({target_mass:.4f})"
            )
            for leaf in _scalable_leaves(node):
                leaf.mass_scale = 1.0
        else:
            try:
                _apply_assembly_mass_override(node, max(target_for_children, 0.0))
                node.step_metadata.pop('mass_error', None)
            except (ValueError, RuntimeError) as exc:
                node.step_metadata['mass_error'] = str(exc)
                for leaf in _scalable_leaves(node):
                    leaf.mass_scale = 1.0
            else:
                # Aggregate children with mass_scale applied
                child_results = []
                for child in node.children:
                    child_results.append(aggregate_properties(child, apply_mass_scale=True))

                total_mass = sum(r.mass for r in child_results)
                total_volume = sum(r.volume for r in child_results)

                # Include node's own geometry
                if own.mass > 0:
                    total_mass += own.mass
                    total_volume += own.volume

                # Verification check: sum must match the override target
                if abs(total_mass - target_mass) > 1e-6:
                    node.step_metadata['mass_error'] = (
                        f"Mass verification failed: expected {target_mass:.6f}, got {total_mass:.6f}"
                    )
                    for leaf in _scalable_leaves(node):
                        leaf.mass_scale = 1.0
                else:
                    if total_mass <= 0:
                        return MassProperties()

                    # Combined CG (weighted average)
                    total_cg_num = np.zeros(3)
                    for child, props in zip(node.children, child_results):
                        cg_local = np.append(props.cg, 1.0)
                        cg_world = (child.instance_transform @ cg_local)[:3]
                        total_cg_num += props.mass * cg_world

                    if own.mass > 0:
                        own_cg_local = np.append(own.cg, 1.0)
                        own_cg_world = (node.instance_transform @ own_cg_local)[:3]
                        total_cg_num += own.mass * own_cg_world

                    combined_cg = total_cg_num / total_mass

                    # Combine inertias with parallel axis theorem
                    total_inertia = np.zeros((3, 3))
                    for child, props in zip(node.children, child_results):
                        cg_local = np.append(props.cg, 1.0)
                        cg_world = (child.instance_transform @ cg_local)[:3]
                        d = cg_world - combined_cg
                        R = child.instance_transform[:3, :3]
                        I_world = R @ props.inertia @ R.T
                        dd = np.dot(d, d)
                        dyad = np.outer(d, d)
                        total_inertia += I_world + props.mass * (dd * np.eye(3) - dyad)

                    if own.mass > 0:
                        own_cg_local = np.append(own.cg, 1.0)
                        own_cg_world = (node.instance_transform @ own_cg_local)[:3]
                        d = own_cg_world - combined_cg
                        R = node.instance_transform[:3, :3]
                        I_world = R @ own.inertia @ R.T
                        dd = np.dot(d, d)
                        dyad = np.outer(d, d)
                        total_inertia += I_world + own.mass * (dd * np.eye(3) - dyad)

                    return MassProperties(
                        mass=total_mass,
                        cg=combined_cg,
                        inertia=total_inertia,
                        volume=total_volume,
                    )

    # Normal aggregation (no override, or override failed / fell back)
    child_results = []
    for child in node.children:
        child_results.append(aggregate_properties(child, apply_mass_scale=True))

    total_mass = sum(r.mass for r in child_results)
    total_volume = sum(r.volume for r in child_results)

    if own.mass > 0:
        total_mass += own.mass
        total_volume += own.volume

    if total_mass <= 0:
        return MassProperties()

    # Combined CG (weighted average)
    total_cg_num = np.zeros(3)
    for child, props in zip(node.children, child_results):
        cg_local = np.append(props.cg, 1.0)
        cg_world = (child.instance_transform @ cg_local)[:3]
        total_cg_num += props.mass * cg_world

    if own.mass > 0:
        own_cg_local = np.append(own.cg, 1.0)
        own_cg_world = (node.instance_transform @ own_cg_local)[:3]
        total_cg_num += own.mass * own_cg_world

    combined_cg = total_cg_num / total_mass

    # Combine inertias with parallel axis theorem
    total_inertia = np.zeros((3, 3))
    for child, props in zip(node.children, child_results):
        cg_local = np.append(props.cg, 1.0)
        cg_world = (child.instance_transform @ cg_local)[:3]
        d = cg_world - combined_cg
        R = child.instance_transform[:3, :3]
        I_world = R @ props.inertia @ R.T
        dd = np.dot(d, d)
        dyad = np.outer(d, d)
        total_inertia += I_world + props.mass * (dd * np.eye(3) - dyad)

    if own.mass > 0:
        own_cg_local = np.append(own.cg, 1.0)
        own_cg_world = (node.instance_transform @ own_cg_local)[:3]
        d = own_cg_world - combined_cg
        R = node.instance_transform[:3, :3]
        I_world = R @ own.inertia @ R.T
        dd = np.dot(d, d)
        dyad = np.outer(d, d)
        total_inertia += I_world + own.mass * (dd * np.eye(3) - dyad)

    return MassProperties(
        mass=total_mass,
        cg=combined_cg,
        inertia=total_inertia,
        volume=total_volume,
    )


def rebalance_all_assembly_overrides(node: Component) -> None:
    """Walk the entire component tree and refresh every assembly mass override.

    Call this after any leaf property change (override, mesh update, etc.) so
    that ancestor assemblies with mass overrides stay locked to their target.
    """
    # Reset all scales first so every override is computed from a clean state
    _reset_all_mass_scales(node)
    _rebalance_post_order(node)


def _rebalance_post_order(node: Component) -> None:
    """Process children first, then the node itself (post-order)."""
    for child in node.children:
        _rebalance_post_order(child)
    if node.children:
        aggregate_properties(node)


def update_component_transform(node: Component, parent_transform: Optional[np.ndarray] = None) -> None:
    """Update the world transform of a node by composing with parent transform.

    Stores result in node.step_metadata['world_transform'].
    """
    if parent_transform is None:
        parent_transform = np.eye(4)
    world = parent_transform @ node.instance_transform
    node.step_metadata["world_transform"] = world
    for child in node.children:
        update_component_transform(child, world)
