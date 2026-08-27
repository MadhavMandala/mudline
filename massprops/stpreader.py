import glob
import os
import subprocess
import tempfile
import gmsh
import numpy as np
import plotly.graph_objects as go
import re
import pint
import trimesh

def extract_step_unit(file_path):
    """Scans the raw ASCII text of the STEP file to detect its native unit."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read(50000) 
        if re.search(r"SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\.\s*\)", content, re.IGNORECASE): return "millimeter"
        elif re.search(r"SI_UNIT\s*\(\s*\.CENTI\.\s*,\s*\.METRE\.\s*\)", content, re.IGNORECASE): return "centimeter"
        elif re.search(r"SI_UNIT\s*\(\s*\$\s*,\s*\.METRE\.\s*\)", content, re.IGNORECASE): return "meter"
        elif re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'INCH'", content, re.IGNORECASE): return "inch"
        elif re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'FOOT'", content, re.IGNORECASE): return "foot"
    return "millimeter"

def load_and_mesh_step(file_path, mesh_size=None, verbose=False):
    """Reads a STEP file, tessellates it, scales to inches natively, and extracts data."""
    native_unit = extract_step_unit(file_path)
    ureg = pint.UnitRegistry()
    to_inch_factor = ureg(native_unit).to(ureg.inch).magnitude
    
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
        
    try:
        gmsh.model.occ.importShapes(file_path)
        gmsh.model.occ.synchronize()
        
        if mesh_size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size)
            
        gmsh.model.mesh.generate(2)
        
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise ValueError("No mesh generated.")
            
        node_to_index = {tag: index for index, tag in enumerate(node_tags)}
        
        x = np.array(node_coords[0::3]) * to_inch_factor
        y = np.array(node_coords[1::3]) * to_inch_factor
        z = np.array(node_coords[2::3]) * to_inch_factor
        
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        
        i_list, j_list, k_list = [], [],[]
        for e_type, e_nodes in zip(elem_types, elem_node_tags):
            if e_type == 2:  
                for n in range(0, len(e_nodes), 3):
                    i_list.append(node_to_index[e_nodes[n]])
                    j_list.append(node_to_index[e_nodes[n+1]])
                    k_list.append(node_to_index[e_nodes[n+2]])
    finally:
        gmsh.finalize()
        
    return x, y, z, i_list, j_list, k_list

def compute_mass_properties(x, y, z, i, j, k, input_mass):
    """
    Computes the Center of Gravity and Inertia Tensor using trimesh.
    """
    # 1. Stack the separated coordinates back into N x 3 arrays for trimesh
    vertices = np.column_stack((x, y, z))
    faces = np.column_stack((i, j, k))
    
    # 2. Build the mesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Check if the surface is fully enclosed (a valid solid body)
    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight (it has tiny gaps). Properties may be slightly approximated.")
        
    # 3. Set the density to achieve the target mass
    # Trimesh derives mass from volume * density. 
    if mesh.volume > 0:
        mesh.density = input_mass / mesh.volume
    else:
        raise ValueError("Error: The calculated volume of the mesh is 0. Is this a 2D surface instead of a solid part?")
    
    # 4. Extract mathematical properties
    cg = mesh.center_mass
    # trimesh evaluates the moment of inertia evaluated at the center of mass automatically.
    inertia_tensor = mesh.moment_inertia 
    
    return cg, inertia_tensor

def _show_figure(fig, title="STEP File Viewer"):
    """Open a Plotly figure in a dedicated standalone window using a browser's app mode."""
    html = fig.to_html(include_plotlyjs='cdn', full_html=True)
    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html)
        path = f.name

    # Look for Chrome/Edge/Brave so we can launch a chromeless app window.
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    ]
    browser = next((c for c in candidates if os.path.exists(c)), None)

    url = f"file:///{path.replace(os.sep, '/')}"
    if browser:
        subprocess.Popen(
            [browser, f"--app={url}", "--window-size=1280,900", "--window-position=100,50"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        import webbrowser
        webbrowser.open(url)


def plot_mesh(x, y, z, i, j, k, cg=None, show_edges=True):
    """Plots the triangular mesh and the CG using Plotly."""
    plot_data =[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='lightgray',
            opacity=0.6, # Lowered opacity slightly so you can see the CG dot inside
            flatshading=show_edges,
            lighting=dict(ambient=0.4, diffuse=0.8, roughness=0.5, specular=0.4, fresnel=0.2),
            lightposition=dict(x=100, y=200, z=500),
            name="Geometry"
        )
    ]
    
    # Add a visual marker for the Center of Gravity
    if cg is not None:
        plot_data.append(
            go.Scatter3d(
                x=[cg[0]], y=[cg[1]], z=[cg[2]],
                mode='markers',
                marker=dict(size=8, color='red', symbol='circle'),
                name='Center of Gravity (CG)'
            )
        )
        
    fig = go.Figure(data=plot_data)
    fig.update_layout(
        title="STEP File Viewer",
        scene=dict(
            aspectmode='data',
            xaxis_title="X (inches)",
            yaxis_title="Y (inches)",
            zaxis_title="Z (inches)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.02, y=0.98)
    )
    _show_figure(fig, title="STEP File Viewer")

_PART_COLORS = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
    '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
]

def build_assembly_tree(folder_path):
    """Parses NAUO entities in each STEP file to build the assembly hierarchy.

    Returns (root, children, name_to_file) where:
      children     — {parent: [virtual_child, ...]}; duplicate children get an @N suffix
      name_to_file — maps every virtual name → the real filename stem for mesh lookup
    """
    children = {}
    name_to_file = {}
    all_original_child_names = set()

    for file_path in glob.glob(os.path.join(folder_path, "*.stp")):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
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

        # Count occurrences so we can give duplicate children a _1, _2 … suffix
        counts = {}
        for n in nauo_names:
            counts[n] = counts.get(n, 0) + 1

        seen = {}
        virtual_children = []
        for n in nauo_names:
            if counts[n] > 1:
                seen[n] = seen.get(n, 0) + 1
                vname = f"{n}@{seen[n]}"
            else:
                vname = n
            virtual_children.append(vname)
            name_to_file[vname] = n          # virtual → real filename stem

        name_to_file[parent] = parent
        children[parent] = virtual_children
        all_original_child_names.update(nauo_names)

    roots = set(children) - all_original_child_names
    root = roots.pop() if roots else next(iter(children), None)
    return root, children, name_to_file


def _vname_to_display(vname):
    """'X1-D551101-01_-@2' → 'X1-D551101-01_2',  'X1-D551071-01_-' → 'X1-D551071-01'"""
    if '@' in vname:
        base, idx = vname.rsplit('@', 1)
        return re.sub(r'_.*$', '', base) + f'_{idx}'
    return re.sub(r'_.*$', '', vname)


def _build_parent_map(node, children_dict, parent=None, result=None):
    """Returns {vname → parent_vname} for every node in the tree."""
    if result is None:
        result = {}
    result[node] = parent
    for child in children_dict.get(node, []):
        _build_parent_map(child, children_dict, node, result)
    return result


def _dfs_order(node, children, depth=0):
    result = [(node, depth)]
    for child in children.get(node, []):
        result.extend(_dfs_order(child, children, depth + 1))
    return result

def load_assembly_folder(folder_path, mesh_size=None, verbose=False):
    """Loads every .stp file in a folder and returns a list of part dicts."""
    stp_files = sorted(glob.glob(os.path.join(folder_path, "*.stp")))
    if not stp_files:
        raise ValueError(f"No .stp files found in: {folder_path}")

    parts = []
    for file_path in stp_files:
        name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Loading {name} ...", end=" ", flush=True)
        try:
            x, y, z, i, j, k = load_and_mesh_step(file_path, mesh_size=mesh_size, verbose=verbose)
            parts.append({"name": name, "x": x, "y": y, "z": z, "i": i, "j": j, "k": k})
            print(f"{len(x)} vertices, {len(i)} triangles")
        except Exception as e:
            print(f"FAILED — {e}")

    return parts

def plot_assembly(parts, cg=None, show_edges=True, tree=None):
    """Plots assembly parts. Pass tree=(root, children_dict, name_to_file) for a
    hierarchical legend where each sub-assembly is a collapsible legendgroup."""
    part_by_name = {p["name"]: p for p in parts}

    if tree is not None:
        root, children_dict, name_to_file = tree
        ordered = _dfs_order(root, children_dict)
        parent_map = _build_parent_map(root, children_dict)
        referenced = set(name_to_file.values())
        for p in parts:
            if p["name"] not in referenced:
                ordered.append((p["name"], 0))
                parent_map[p["name"]] = None
                name_to_file[p["name"]] = p["name"]
    else:
        ordered = [(p["name"], 0) for p in parts]
        name_to_file = {p["name"]: p["name"] for p in parts}
        parent_map = {p["name"]: None for p in parts}

    plot_data = []
    for color_idx, (vname, depth) in enumerate(ordered):
        file_stem = name_to_file.get(vname, vname)
        part = part_by_name.get(file_stem)
        if part is None:
            continue

        display = _vname_to_display(vname)
        parent_vname = parent_map.get(vname)
        if parent_vname is not None:
            lg = _vname_to_display(parent_vname)
            lg_title = dict(text=lg)
        else:
            lg = display
            lg_title = None

        plot_data.append(
            go.Mesh3d(
                x=part["x"], y=part["y"], z=part["z"],
                i=part["i"], j=part["j"], k=part["k"],
                color=_PART_COLORS[color_idx % len(_PART_COLORS)],
                opacity=0.7,
                flatshading=show_edges,
                lighting=dict(ambient=0.4, diffuse=0.8, roughness=0.5, specular=0.4, fresnel=0.2),
                lightposition=dict(x=100, y=200, z=500),
                name=display,
                showlegend=False,
            )
        )

    if cg is not None:
        plot_data.append(
            go.Scatter3d(
                x=[cg[0]], y=[cg[1]], z=[cg[2]],
                mode='markers',
                marker=dict(size=8, color='red', symbol='circle'),
                name='Center of Gravity (CG)'
            )
        )

    fig = go.Figure(data=plot_data)
    fig.update_layout(
        title="Assembly Viewer",
        scene=dict(
            aspectmode='data',
            xaxis_title="X (inches)",
            yaxis_title="Y (inches)",
            zaxis_title="Z (inches)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
    )
    _show_figure(fig, title="Assembly Viewer")

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    resolution = 0.5  # 1.0 = default, 0.5 = finer, 2.0 = coarser

    # --- Single file ---
    # step_file = "X1-00C01_-2.stp"
    # part_mass = 1000.0
    # x, y, z, i, j, k = load_and_mesh_step(step_file, mesh_size=resolution)
    # cg, inertia = compute_mass_properties(x, y, z, i, j, k, input_mass=part_mass)
    # print(f"CG: {np.round(cg, 4)}\nInertia:\n{np.round(inertia, 4)}")
    # plot_mesh(x, y, z, i, j, k, cg=cg)

    # --- Assembly folder ---
    assembly_folder = os.path.join(os.path.dirname(__file__), "step_test")
    try:
        parts = load_assembly_folder(assembly_folder, mesh_size=resolution)
        tree = build_assembly_tree(assembly_folder)
        print(f"\nLoaded {len(parts)} parts.")
        plot_assembly(parts, tree=tree)  # tree is the 3-tuple (root, children_dict, name_to_file)
    except Exception as e:
        print(f"Error: {e}")