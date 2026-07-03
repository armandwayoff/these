import meshio
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python gmsh_to_pgf3d.py mesh.msh")
    sys.exit(1)

filename = sys.argv[1]
mesh = meshio.read(filename)
points = mesh.points[:, :3]

# --- Collect surface faces ---
faces = []  # list of list of point indices

def add_triangle(a, b, c):
    faces.append([a, b, c])

def add_quad(a, b, c, d):
    faces.append([a, b, c, d])

for cellblock in mesh.cells:
    t = cellblock.type
    for cell in cellblock.data:
        if t == "triangle":
            add_triangle(cell[0], cell[1], cell[2])
        elif t == "quad":
            add_quad(cell[0], cell[1], cell[2], cell[3])
        # Faces des volumes
        elif t == "tetra":
            add_triangle(cell[0], cell[1], cell[2])
            add_triangle(cell[0], cell[1], cell[3])
            add_triangle(cell[0], cell[2], cell[3])
            add_triangle(cell[1], cell[2], cell[3])
        elif t == "hexahedron":
            add_quad(cell[0], cell[1], cell[2], cell[3])
            add_quad(cell[4], cell[5], cell[6], cell[7])
            add_quad(cell[0], cell[1], cell[5], cell[4])
            add_quad(cell[1], cell[2], cell[6], cell[5])
            add_quad(cell[2], cell[3], cell[7], cell[6])
            add_quad(cell[3], cell[0], cell[4], cell[7])
        elif t == "wedge":
            add_triangle(cell[0], cell[1], cell[2])
            add_triangle(cell[3], cell[4], cell[5])
            add_quad(cell[0], cell[1], cell[4], cell[3])
            add_quad(cell[1], cell[2], cell[5], cell[4])
            add_quad(cell[2], cell[0], cell[3], cell[5])
        elif t == "pyramid":
            add_quad(cell[0], cell[1], cell[2], cell[3])
            add_triangle(cell[0], cell[1], cell[4])
            add_triangle(cell[1], cell[2], cell[4])
            add_triangle(cell[2], cell[3], cell[4])
            add_triangle(cell[3], cell[0], cell[4])

# --- Write faces file (triangles only, quads split into 2 triangles) ---
with open("faces3d.dat", "w") as f:
    f.write("x y z\n")
    for face in faces:
        if len(face) == 3:
            tris = [face]
        else:  # quad → 2 triangles
            tris = [
                [face[0], face[1], face[2]],
                [face[0], face[2], face[3]],
            ]
        for tri in tris:
            for idx in tri:
                p = points[idx]
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
            f.write("\n")

print("Written faces3d.dat")