"""Generate simple STEP files for testing MassProp."""
import gmsh
from pathlib import Path

OUT_DIR = Path(__file__).parent

def make_cube(filename, size=10.0):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cube")
    gmsh.model.occ.addBox(0, 0, 0, size, size, size)
    gmsh.model.occ.synchronize()
    gmsh.write(str(OUT_DIR / filename))
    gmsh.finalize()
    print(f"Created {filename} ({size} mm cube)")

def make_cylinder(filename, radius=5.0, height=20.0):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cylinder")
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, radius)
    gmsh.model.occ.synchronize()
    gmsh.write(str(OUT_DIR / filename))
    gmsh.finalize()
    print(f"Created {filename} (r={radius}, h={height} mm cylinder)")

def make_sphere(filename, radius=5.0):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("sphere")
    gmsh.model.occ.addSphere(0, 0, 0, radius)
    gmsh.model.occ.synchronize()
    gmsh.write(str(OUT_DIR / filename))
    gmsh.finalize()
    print(f"Created {filename} (r={radius} mm sphere)")

if __name__ == "__main__":
    make_cube("cube_10mm.step", size=10.0)
    make_cube("cube_50mm.step", size=50.0)
    make_cylinder("cylinder_5x20mm.step", radius=5.0, height=20.0)
    make_sphere("sphere_10mm.step", radius=10.0)
