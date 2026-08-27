"""Material library.

Densities are the one property every downstream consumer needs: CAD gives
volume, and volume times density is the mass that mass properties, trajectory
and structural margin all depend on. Keeping them in one table means a material
substitution is a single edit rather than a search for hardcoded numbers.

Densities are nominal handbook values in kg/m^3. Composite values in particular
depend on layup and fibre volume fraction, so they should be replaced with
measured coupon values before anything is built to them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """A structural material.

    Args:
        name: Identifier used in vehicle definitions.
        density_kg_m3: Bulk density.
        yield_strength_pa: Tensile yield. Zero where not meaningful (or not
            yet filled in) -- structural checks must treat zero as "unknown"
            and refuse to compute a margin rather than report a false one.
        youngs_modulus_pa: Elastic modulus, needed for buckling and flutter.
        shear_modulus_pa: Needed for fin flutter, which is torsion-driven.
    """

    name: str
    density_kg_m3: float
    yield_strength_pa: float = 0.0
    youngs_modulus_pa: float = 0.0
    shear_modulus_pa: float = 0.0
    #: How the material looks, as linear RGB in 0..1.
    #:
    #: Appearance lives on the material rather than in the viewport because it
    #: is a property of the material: swapping a tube from fibreglass to carbon
    #: should change what it weighs *and* what it looks like, from one edit. A
    #: colour table in the renderer would have to be kept in step with this one
    #: by hand, and would know nothing about a material imported from a STEP
    #: file at runtime.
    color: tuple[float, float, float] = (0.72, 0.76, 0.82)
    #: 0 is a matte composite, 1 a polished metal. Drives the specular
    #: highlight, which is what actually distinguishes aluminium from carbon on
    #: screen -- colour alone leaves both reading as flat plastic.
    sheen: float = 0.4

    @property
    def density_lbm_in3(self) -> float:
        """Density in massprops' internal units."""
        return self.density_kg_m3 * 3.61273e-5


MATERIALS: dict[str, Material] = {
    "aluminium_6061_t6": Material(
        "aluminium_6061_t6", 2700.0,
        yield_strength_pa=276e6, youngs_modulus_pa=68.9e9, shear_modulus_pa=26.0e9,
        color=(0.78, 0.80, 0.83), sheen=0.80,
    ),
    "aluminium_7075_t6": Material(
        "aluminium_7075_t6", 2810.0,
        yield_strength_pa=503e6, youngs_modulus_pa=71.7e9, shear_modulus_pa=26.9e9,
        color=(0.73, 0.75, 0.79), sheen=0.82,
    ),
    "steel_4130": Material(
        "steel_4130", 7850.0,
        yield_strength_pa=460e6, youngs_modulus_pa=205e9, shear_modulus_pa=80e9,
        color=(0.52, 0.55, 0.60), sheen=0.90,
    ),
    "stainless_304": Material(
        "stainless_304", 8000.0,
        yield_strength_pa=215e6, youngs_modulus_pa=193e9, shear_modulus_pa=77e9,
        color=(0.83, 0.85, 0.88), sheen=1.00,
    ),
    "cfrp_quasi_isotropic": Material(
        "cfrp_quasi_isotropic", 1600.0,
        yield_strength_pa=600e6, youngs_modulus_pa=70e9, shear_modulus_pa=26e9,
        # Near black with a resin gloss, which is what a cured laminate is.
        color=(0.15, 0.16, 0.18), sheen=0.55,
    ),
    "g10_fiberglass": Material(
        "g10_fiberglass", 1800.0,
        yield_strength_pa=262e6, youngs_modulus_pa=18.6e9, shear_modulus_pa=7.0e9,
        # G10's characteristic olive-tan.
        color=(0.63, 0.60, 0.36), sheen=0.30,
    ),
    "phenolic": Material(
        "phenolic", 1350.0,
        yield_strength_pa=55e6, youngs_modulus_pa=6.0e9, shear_modulus_pa=2.2e9,
        color=(0.40, 0.29, 0.21), sheen=0.15,
    ),
    "titanium_6al4v": Material(
        "titanium_6al4v", 4430.0,
        yield_strength_pa=880e6, youngs_modulus_pa=113.8e9, shear_modulus_pa=44e9,
        color=(0.66, 0.64, 0.62), sheen=0.70,
    ),
}


def register_material(material: Material, replace: bool = False) -> Material:
    """Add a material discovered at runtime, e.g. read out of a STEP file.

    Imported CAD names its own materials, and they will not be in the table
    above. Registering rather than special-casing keeps one concept -- a part
    has a material name, that name has a density -- so a vehicle carrying an
    imported part still serialises to a name and reloads correctly.

    An existing name is kept unless ``replace``, since the curated entries here
    carry strength and modulus that an imported density does not.
    """
    existing = MATERIALS.get(material.name)
    if existing is not None and not replace:
        return existing
    MATERIALS[material.name] = material
    return material


def material_named(density_kg_m3: float, name: str = "") -> Material:
    """A material for an imported density, registered under a safe name."""
    safe = "".join(
        c if c.isalnum() else "_" for c in (name or "imported").strip().lower()
    ).strip("_") or "imported"
    if safe in MATERIALS and abs(MATERIALS[safe].density_kg_m3 - density_kg_m3) > 1e-9:
        safe = f"{safe}_{density_kg_m3:.0f}"
    return register_material(Material(safe, float(density_kg_m3)))


def get_material(name: str) -> Material:
    """Look up a material, failing loudly on a typo.

    A silent default here would put a wrong density into every mass number
    downstream, so an unknown name is an error rather than a fallback.
    """
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(
            f"Unknown material {name!r}. Known materials: "
            f"{', '.join(sorted(MATERIALS))}"
        ) from None
