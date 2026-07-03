import meshio
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python gmsh_to_pgf3d.py mesh.msh")
    sys.exit(1)

filename = sys.argv[1]
mesh = meshio.read(filename)

points = mesh.points[:, :3]  # (x,y,z)

# --- Extract unique edges ---
edges = set()

def add_edge(a, b):
    if a < b:
        edges.add((a, b))
    else:
        edges.add((b, a))

# Element → edges mapping
def extract_edges_from_cell(type_name, cell):
    if type_name == "line":
        add_edge(cell[0], cell[1])

    # triangles (surface)
    if type_name == "triangle":
        add_edge(cell[0], cell[1])
        add_edge(cell[1], cell[2])
        add_edge(cell[2], cell[0])

    # quads (surface)
    if type_name == "quad":
        add_edge(cell[0], cell[1])
        add_edge(cell[1], cell[2])
        add_edge(cell[2], cell[3])
        add_edge(cell[3], cell[0])

    # tetra (4-node)
    if type_name == "tetra":
        add_edge(cell[0], cell[1])
        add_edge(cell[0], cell[2])
        add_edge(cell[0], cell[3])
        add_edge(cell[1], cell[2])
        add_edge(cell[1], cell[3])
        add_edge(cell[2], cell[3])

    # hexa (8-node)
    if type_name == "hexahedron":
        # bottom face
        add_edge(cell[0], cell[1])
        add_edge(cell[1], cell[2])
        add_edge(cell[2], cell[3])
        add_edge(cell[3], cell[0])
        # top face
        add_edge(cell[4], cell[5])
        add_edge(cell[5], cell[6])
        add_edge(cell[6], cell[7])
        add_edge(cell[7], cell[4])
        # verticals
        add_edge(cell[0], cell[4])
        add_edge(cell[1], cell[5])
        add_edge(cell[2], cell[6])
        add_edge(cell[3], cell[7])

    # prism (6 nodes)
    if type_name == "wedge":
        # bottom triangle
        add_edge(cell[0], cell[1])
        add_edge(cell[1], cell[2])
        add_edge(cell[2], cell[0])
        # top triangle
        add_edge(cell[3], cell[4])
        add_edge(cell[4], cell[5])
        add_edge(cell[5], cell[3])
        # vertical edges
        add_edge(cell[0], cell[3])
        add_edge(cell[1], cell[4])
        add_edge(cell[2], cell[5])

    # pyramid (5 nodes)
    if type_name == "pyramid":
        # base quad
        add_edge(cell[0], cell[1])
        add_edge(cell[1], cell[2])
        add_edge(cell[2], cell[3])
        add_edge(cell[3], cell[0])
        # top edges (apex = 4)
        add_edge(cell[0], cell[4])
        add_edge(cell[1], cell[4])
        add_edge(cell[2], cell[4])
        add_edge(cell[3], cell[4])


# Parse all cell blocks
for cellblock in mesh.cells:
    for elem in cellblock.data:
        extract_edges_from_cell(cellblock.type, elem)

# --- Write nodes file ---
with open("nodes3d.dat", "w") as f:
    f.write("x y z\n")
    for p in points:
        f.write(f"{p[0]} {p[1]} {p[2]}\n")

# --- Write edges file ---
with open("edges3d.dat", "w") as f:
    f.write("x y z\n")
    for (i, j) in edges:
        p1 = points[i]
        p2 = points[j]
        f.write(f"{p1[0]} {p1[1]} {p1[2]}\n")
        f.write(f"{p2[0]} {p2[1]} {p2[2]}\n")
        f.write("\n")

print("Written nodes3d.dat and edges3d.dat")
