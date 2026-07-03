from pathlib import Path
import numpy as np
import math

inp = Path("../data/faces3D_sphere.dat")
out_arcs= Path("maillage_sphere_arcs_avant.dat")
out_arcs_diamants= Path("maillage_sphere_arcs_diamants_avant.dat")
out_bary = Path("maillage_sphere_barycentres_avant.dat")
out_milieux = Path("maillage_sphere_milieux_avant.dat")
out_dual = Path("maillage_sphere_dual_avant.dat")
out_dual_polyedrique = Path("maillage_sphere_dual_polyedrique_avant.dat")

N = 20

az = math.radians(-10)
el = math.radians(22)

view_dir = np.array([
    math.cos(az) * math.cos(el),
    math.sin(az) * math.cos(el),
    math.sin(el),
])

sign_view = 1  # essayer 1 si l'hémisphère est inversé

def normalize(P):
    P = np.array(P, dtype=float)
    return P / np.linalg.norm(P)

def visible_point(P):
    P = normalize(P)
    return sign_view * np.dot(P, view_dir) > 0

def visible_triangle(A, B, C):
    return visible_point(A) and visible_point(B) and visible_point(C)

def arc_points(A, B, n=N):
    A = normalize(A)
    B = normalize(B)

    for t in np.linspace(0, 1, n):
        P = normalize((1 - t) * A + t * B)
        yield P

def write_arc(g, A, B):
    for P in arc_points(A, B):
        g.write(f"{P[0]} {P[1]} {P[2]}\n")
    g.write("\n")

def write_segment(g, P, Q):
    g.write(f"{P[0]} {P[1]} {P[2]}\n")
    g.write(f"{Q[0]} {Q[1]} {Q[2]}\n")
    g.write("\n")

with inp.open() as f, out_arcs.open("w") as g:
    next(f)  # ignore l'en-tête : x y z

    pts = []

    for line in f:
        if line.strip():
            x, y, z = map(float, line.split())
            pts.append((x, y, z))

            if len(pts) == 3:
                A, B, C = pts

                if visible_triangle(A, B, C):
                    write_arc(g, A, B)
                    write_arc(g, B, C)
                    write_arc(g, C, A)

                pts = []

with inp.open() as f, out_bary.open("w") as g:
    next(f)  # ignore l'en-tête : x y z

    pts = []

    for line in f:
        if line.strip():
            x, y, z = map(float, line.split())
            pts.append(np.array([x, y, z]))

            if len(pts) == 3:
                A, B, C = pts

                if visible_triangle(A, B, C):
                    G = (A + B + C) / 3
                    g.write(f"{G[0]} {G[1]} {G[2]}\n")

                pts = []

with inp.open() as f, out_milieux.open("w") as g:
    next(f)  # ignore l'en-tête : x y z

    pts = []

    for line in f:
        if line.strip():
            x, y, z = map(float, line.split())
            pts.append(np.array([x, y, z]))

            if len(pts) == 3:
                A, B, C = pts

                if visible_triangle(A, B, C):
                    M_AB = (A + B) / 2
                    M_BC = (B + C) / 2
                    M_CA = (C + A) / 2

                    g.write(f"{M_AB[0]} {M_AB[1]} {M_AB[2]}\n")
                    g.write(f"{M_BC[0]} {M_BC[1]} {M_BC[2]}\n")
                    g.write(f"{M_CA[0]} {M_CA[1]} {M_CA[2]}\n")

                pts = []


with inp.open() as f, out_dual.open("w") as g:
    next(f)

    pts = []

    for line in f:
        if line.strip():
            pts.append(np.array(list(map(float, line.split()))))

            if len(pts) == 3:
                A, B, C = pts

                if visible_triangle(A, B, C):
                    G = normalize((A + B + C) / 3)

                    M_AB = normalize((A + B) / 2)
                    M_BC = normalize((B + C) / 2)
                    M_CA = normalize((C + A) / 2)

                    write_arc(g, G, M_AB)
                    write_arc(g, G, M_BC)
                    write_arc(g, G, M_CA)

                pts = []
    

with inp.open() as f, out_dual_polyedrique.open("w") as g:
    next(f)  # ignore l'en-tête : x y z

    pts = []

    for line in f:
        if line.strip():
            pts.append(np.array(list(map(float, line.split()))))

            if len(pts) == 3:
                A, B, C = pts

                if visible_triangle(A, B, C):
                    G = (A + B + C) / 3

                    M_AB = (A + B) / 2
                    M_BC = (B + C) / 2
                    M_CA = (C + A) / 2

                    write_segment(g, G, M_AB)
                    write_segment(g, G, M_BC)
                    write_segment(g, G, M_CA)

                    write_segment(g, G, A)
                    write_segment(g, G, B)
                    write_segment(g, G, C)

                pts = []


with inp.open() as f, out_arcs_diamants.open("w") as g:
    next(f)

    pts = []

    for line in f:
        if line.strip():
            pts.append(np.array(list(map(float, line.split()))))

            if len(pts) == 3:
                A, B, C = pts

                if visible_triangle(A, B, C):
                    G = normalize((A + B + C) / 3)

                    write_arc(g, G, A)
                    write_arc(g, G, B)
                    write_arc(g, G, C)

                pts = []