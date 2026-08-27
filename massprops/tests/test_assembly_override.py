import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from massprops.model.models import Component, MassProperties
from massprops.model.assembly import aggregate_properties, _leaf_descendants, rebalance_all_assembly_overrides


def test_assembly_mass_override_rebalances_on_change():
    """If an assembly has a mass override, adding/changing children should
    dynamically rebalance leaf mass_scale so the total stays at the override."""
    # Build a simple assembly: two leaves each with mass 10
    root = Component(name="Assembly")
    leaf_a = Component(name="LeafA")
    leaf_a.computed_props = MassProperties(mass=10.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    leaf_b = Component(name="LeafB")
    leaf_b.computed_props = MassProperties(mass=10.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    root.children = [leaf_a, leaf_b]

    # Apply a mass override: target mass = 30 (scale should be 1.5)
    root.overridden_props = MassProperties(mass=30.0, cg=np.zeros(3), inertia=np.eye(3), volume=20.0)
    root.override_fields = {'mass'}

    # First aggregation should set scales and return the override mass
    props = aggregate_properties(root)
    assert props.mass == 30.0, f"Expected 30.0, got {props.mass}"
    assert leaf_a.mass_scale == 1.5
    assert leaf_b.mass_scale == 1.5

    # Now add a third leaf with mass 10 (raw total becomes 30, scale should become 1.0)
    leaf_c = Component(name="LeafC")
    leaf_c.computed_props = MassProperties(mass=10.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    root.children.append(leaf_c)

    props = aggregate_properties(root)
    assert props.mass == 30.0, f"Expected 30.0 after adding child, got {props.mass}"
    assert leaf_a.mass_scale == 1.0
    assert leaf_b.mass_scale == 1.0
    assert leaf_c.mass_scale == 1.0

    # Change leaf_a's computed mass to 20 (raw total becomes 40, scale should become 0.75)
    leaf_a.computed_props = MassProperties(mass=20.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    props = aggregate_properties(root)
    assert props.mass == 30.0, f"Expected 30.0 after changing child, got {props.mass}"
    assert leaf_a.mass_scale == 0.75
    assert leaf_b.mass_scale == 0.75
    assert leaf_c.mass_scale == 0.75


def test_assembly_override_cleared_resets_scales():
    """Clearing an assembly mass override should reset leaf mass_scale to 1.0."""
    root = Component(name="Assembly")
    leaf = Component(name="Leaf")
    leaf.computed_props = MassProperties(mass=10.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    root.children = [leaf]

    root.overridden_props = MassProperties(mass=5.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    root.override_fields = {'mass'}

    aggregate_properties(root)
    assert leaf.mass_scale == 0.5

    # Clear override
    root.overridden_props = None
    root.override_fields = set()

    # Rebalance from the root so stale scales are reset
    rebalance_all_assembly_overrides(root)

    props = aggregate_properties(root)
    assert props.mass == 10.0, f"Expected 10.0, got {props.mass}"
    assert leaf.mass_scale == 1.0


def test_leaf_descendants():
    root = Component(name="Root")
    a = Component(name="A")
    b = Component(name="B")
    c = Component(name="C")
    root.children = [a, b]
    a.children = [c]

    leaves = _leaf_descendants(root)
    assert leaves == [c, b]


def test_leaf_override_triggers_ancestor_rebalance():
    """Changing a leaf's mass (via override) should force ancestor assemblies
    with mass overrides to rebalance and hold their target.
    The overridden leaf should keep its exact mass; only the other leaves scale."""
    root = Component(name="Assembly")
    leaf_a = Component(name="LeafA")
    leaf_a.computed_props = MassProperties(mass=10.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    leaf_b = Component(name="LeafB")
    leaf_b.computed_props = MassProperties(mass=10.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    root.children = [leaf_a, leaf_b]

    # Lock assembly mass at 20 (scale = 1.0)
    root.overridden_props = MassProperties(mass=20.0, cg=np.zeros(3), inertia=np.eye(3), volume=20.0)
    root.override_fields = {'mass'}
    aggregate_properties(root)
    assert root.effective_props().mass == 20.0
    assert leaf_a.mass_scale == 1.0
    assert leaf_b.mass_scale == 1.0

    # Now override leaf_a mass to 30. Raw total of flexible leaf_b is 10.
    # Remaining budget = 20 - 30 = -10 -> INFEASIBLE.
    leaf_a.overridden_props = MassProperties(mass=30.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    leaf_a.override_fields = {'mass'}

    # This should raise an error in aggregate_properties and store it in metadata
    props = aggregate_properties(root)
    assert 'mass_error' in root.step_metadata
    # Fallback returns raw total (40)
    assert props.mass == 40.0, f"Expected fallback 40.0, got {props.mass}"

    # Now change leaf_a override to 5. Raw flexible = 10, remaining = 15, scale = 1.5
    leaf_a.overridden_props = MassProperties(mass=5.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    leaf_a.override_fields = {'mass'}
    root.step_metadata.pop('mass_error', None)

    rebalance_all_assembly_overrides(root)

    assert root.effective_props().mass == 20.0, f"Expected 20.0, got {root.effective_props().mass}"
    assert leaf_a.mass_scale == 1.0
    assert leaf_b.mass_scale == 1.5

    # leaf_a's effective mass should be exactly 5 (its override, not scaled)
    assert leaf_a.effective_props().mass == 5.0, f"Expected 5.0, got {leaf_a.effective_props().mass}"
    # leaf_b's effective mass should be 10 * 1.5 = 15
    assert leaf_b.effective_props().mass == 15.0, f"Expected 15.0, got {leaf_b.effective_props().mass}"


def test_fixed_leaf_exact_match():
    """A leaf with its own override should keep exactly that mass, and the
    flexible leaves should absorb the difference."""
    root = Component(name="Assembly")
    fixed = Component(name="Fixed")
    fixed.computed_props = MassProperties(mass=10.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    flex1 = Component(name="Flex1")
    flex1.computed_props = MassProperties(mass=20.0, cg=np.zeros(3), inertia=np.eye(3), volume=20.0)
    flex2 = Component(name="Flex2")
    flex2.computed_props = MassProperties(mass=30.0, cg=np.zeros(3), inertia=np.eye(3), volume=30.0)
    root.children = [fixed, flex1, flex2]

    # fixed leaf override = 15
    fixed.overridden_props = MassProperties(mass=15.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    fixed.override_fields = {'mass'}

    # Assembly target = 60. Flexible raw = 50. Remaining = 45. Scale = 45/50 = 0.9
    root.overridden_props = MassProperties(mass=60.0, cg=np.zeros(3), inertia=np.eye(3), volume=60.0)
    root.override_fields = {'mass'}

    rebalance_all_assembly_overrides(root)

    assert root.effective_props().mass == 60.0
    assert fixed.effective_props().mass == 15.0
    assert flex1.effective_props().mass == 18.0  # 20 * 0.9
    assert flex2.effective_props().mass == 27.0  # 30 * 0.9


def test_nested_assembly_overrides():
    """Parent and child assemblies both with mass overrides should stay independent.
    The parent should treat the child assembly as a fixed black box."""
    root = Component(name="Root")
    sub = Component(name="Sub")
    leaf_a = Component(name="LeafA")
    leaf_a.computed_props = MassProperties(mass=20.0, cg=np.zeros(3), inertia=np.eye(3), volume=20.0)
    leaf_b = Component(name="LeafB")
    leaf_b.computed_props = MassProperties(mass=20.0, cg=np.zeros(3), inertia=np.eye(3), volume=20.0)
    leaf_c = Component(name="LeafC")
    leaf_c.computed_props = MassProperties(mass=20.0, cg=np.zeros(3), inertia=np.eye(3), volume=20.0)

    sub.children = [leaf_a, leaf_b]
    root.children = [sub, leaf_c]

    # SubAssembly target = 30 (scale = 0.75)
    sub.overridden_props = MassProperties(mass=30.0, cg=np.zeros(3), inertia=np.eye(3), volume=40.0)
    sub.override_fields = {'mass'}

    # Root target = 100. Sub fixed = 30. leaf_c flexible raw = 20. scale = 70/20 = 3.5
    root.overridden_props = MassProperties(mass=100.0, cg=np.zeros(3), inertia=np.eye(3), volume=60.0)
    root.override_fields = {'mass'}

    rebalance_all_assembly_overrides(root)

    assert sub.effective_props().mass == 30.0
    assert leaf_a.effective_props().mass == 15.0
    assert leaf_b.effective_props().mass == 15.0
    assert leaf_c.effective_props().mass == 70.0
    assert root.effective_props().mass == 100.0


def test_assembly_with_own_geometry_override():
    """An assembly with its own geometry should subtract that geometry mass from
    the override target before distributing the remainder to children."""
    root = Component(name="Root")
    root.computed_props = MassProperties(mass=5.0, cg=np.zeros(3), inertia=np.eye(3), volume=5.0)
    leaf_a = Component(name="LeafA")
    leaf_a.computed_props = MassProperties(mass=20.0, cg=np.zeros(3), inertia=np.eye(3), volume=20.0)
    leaf_b = Component(name="LeafB")
    leaf_b.computed_props = MassProperties(mass=30.0, cg=np.zeros(3), inertia=np.eye(3), volume=30.0)
    root.children = [leaf_a, leaf_b]

    # Root target = 60. Own geometry = 5. Target for children = 55.
    # Flexible raw = 50. Scale = 55/50 = 1.1
    root.overridden_props = MassProperties(mass=60.0, cg=np.zeros(3), inertia=np.eye(3), volume=60.0)
    root.override_fields = {'mass'}

    rebalance_all_assembly_overrides(root)

    assert root.effective_props().mass == 60.0
    assert leaf_a.effective_props().mass == 22.0  # 20 * 1.1
    assert leaf_b.effective_props().mass == 33.0  # 30 * 1.1


def test_rebalance_idempotent():
    """Calling aggregate_properties multiple times without changes should keep
    the same scales and same total mass."""
    root = Component(name="Assembly")
    leaf_a = Component(name="LeafA")
    leaf_a.computed_props = MassProperties(mass=10.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    leaf_b = Component(name="LeafB")
    leaf_b.computed_props = MassProperties(mass=10.0, cg=np.zeros(3), inertia=np.eye(3), volume=10.0)
    root.children = [leaf_a, leaf_b]

    root.overridden_props = MassProperties(mass=25.0, cg=np.zeros(3), inertia=np.eye(3), volume=20.0)
    root.override_fields = {'mass'}

    rebalance_all_assembly_overrides(root)
    first_scale = leaf_a.mass_scale
    first_mass = root.effective_props().mass

    # Call again directly (not through rebalance)
    props = aggregate_properties(root)
    assert props.mass == first_mass, f"Mass changed on re-aggregation: {first_mass} -> {props.mass}"
    assert leaf_a.mass_scale == first_scale, f"Scale changed on re-aggregation: {first_scale} -> {leaf_a.mass_scale}"
    assert leaf_b.mass_scale == first_scale


if __name__ == "__main__":
    test_assembly_mass_override_rebalances_on_change()
    print("test_assembly_mass_override_rebalances_on_change PASSED")

    test_assembly_override_cleared_resets_scales()
    print("test_assembly_override_cleared_resets_scales PASSED")

    test_leaf_descendants()
    print("test_leaf_descendants PASSED")

    test_leaf_override_triggers_ancestor_rebalance()
    print("test_leaf_override_triggers_ancestor_rebalance PASSED")

    test_fixed_leaf_exact_match()
    print("test_fixed_leaf_exact_match PASSED")

    test_nested_assembly_overrides()
    print("test_nested_assembly_overrides PASSED")

    test_assembly_with_own_geometry_override()
    print("test_assembly_with_own_geometry_override PASSED")

    test_rebalance_idempotent()
    print("test_rebalance_idempotent PASSED")

    print("All tests passed!")
