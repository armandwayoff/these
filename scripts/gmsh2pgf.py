import meshio
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python gmsh2pgf.py mesh.msh")
    sys.exit(1)

filename = sys.argv[1]

mesh = meshio.read(filename)
points = mesh.points[:, :2]  # assume 2D mesh (ignore z)

# --- Extract unique edges ---
edges = set()

def add_edge(a, b):
    if a < b:
        edges.add((a, b))
    else:
        edges.add((b, a))

for cell_block in mesh.cells:
    ctype = cell_block.type
    data = cell_block.data

    if ctype == "line":
        for e in data:
            add_edge(e[0], e[1])

    if ctype == "triangle":
        for tri in data:
            add_edge(tri[0], tri[1])
            add_edge(tri[1], tri[2])
            add_edge(tri[2], tri[0])

    if ctype == "quad":
        for q in data:
            add_edge(q[0], q[1])
            add_edge(q[1], q[2])
            add_edge(q[2], q[3])
            add_edge(q[3], q[0])

# --- Write nodes.dat ---
with open("test/nodes.dat", "w") as f:
    f.write("x y\n")
    for p in points:
        f.write(f"{p[0]} {p[1]}\n")

P = np.array(points)
print(max(P[:,1]))

# --- Write edges.dat ---
with open("test/edges.dat", "w") as f:
    f.write("x y\n")
    for (i, j) in edges:
        p1 = points[i]
        p2 = points[j]
        f.write(f"{p1[0]} {p1[1]}\n")
        f.write(f"{p2[0]} {p2[1]}\n")
        f.write("\n")   # blank line separates segments

print("Written nodes.dat and edges.dat")
