from pathlib import Path
import numpy as np
import math

inp = Path("../data/faces3D_tore_complet.dat")

out_arcs = Path("maillage_tore_arcs_avant.dat")
out_bary = Path("maillage_tore_barycentres_avant.dat")
out_milieux = Path("maillage_tore_milieux_avant.dat")
out_dual_polyedrique = Path("maillage_tore_dual_polyedrique_avant.dat")

az = math.radians(-10)
el = math.radians(22)

view_dir = np.array([
    math.cos(az) * math.cos(el),
    math.sin(az) * math.cos(el),
    math.sin(el),
])

sign_view = 1

"""
def visible_triangle(A, B, C):
    # Critère de visibilité par orientation :
    # on garde les triangles dont la normale est tournée vers le point de vue.

    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    normal = np.cross(B - A, C - A)

    if np.linalg.norm(normal) == 0:
        return False

    normal = normal / np.linalg.norm(normal)

    return sign_view * np.dot(normal, view_dir) > 0
"""

def write_segment(g, P, Q):
    g.write(f"{P[0]} {P[1]} {P[2]}\n")
    g.write(f"{Q[0]} {Q[1]} {Q[2]}\n")
    g.write("\n")

def read_triangles(filename):
    triangles = []

    with filename.open() as f:
        next(f)  # ignore x y z

        pts = []

        for line in f:
            if line.strip():
                pts.append(np.array(list(map(float, line.split()))))

                if len(pts) == 3:
                    triangles.append(tuple(pts))
                    pts = []

    return triangles


R = 1.0
r = 0.3
N = 10

az = math.radians(-10)
el = math.radians(22)

view_dir = np.array([
    math.cos(az) * math.cos(el),
    math.sin(az) * math.cos(el),
    math.sin(el),
])

sign_view = 1

def torus_normal(P):
    x, y, z = P
    rho = math.sqrt(x*x + y*y)

    center = np.array([R * x / rho, R * y / rho, 0.0])
    n = np.array(P) - center
    return n / np.linalg.norm(n)

def visible_point(P):
    return sign_view * np.dot(torus_normal(P), view_dir) > 0

def visible_triangle(A, B, C):
    return visible_point(A) and visible_point(B) and visible_point(C)

def torus_parameters(P):
    x, y, z = P
    u = math.atan2(y, x)

    rho = math.sqrt(x*x + y*y)
    v = math.atan2(z, rho - R)

    return u, v

def unwrap_angle(b, a):
    """
    Remplace b par l'angle équivalent le plus proche de a.
    """
    while b - a > math.pi:
        b -= 2 * math.pi
    while b - a < -math.pi:
        b += 2 * math.pi
    return b

def torus_point(u, v):
    return np.array([
        (R + r * math.cos(v)) * math.cos(u),
        (R + r * math.cos(v)) * math.sin(u),
        r * math.sin(v),
    ])

def arc_points_torus(A, B, n=N):
    uA, vA = torus_parameters(A)
    uB, vB = torus_parameters(B)

    uB = unwrap_angle(uB, uA)
    vB = unwrap_angle(vB, vA)

    for t in np.linspace(0, 1, n):
        u = (1 - t) * uA + t * uB
        v = (1 - t) * vA + t * vB
        yield torus_point(u, v)

def write_arc(g, A, B):
    for P in arc_points_torus(A, B):
        g.write(f"{P[0]} {P[1]} {P[2]}\n")
    g.write("\n")

triangles = read_triangles(inp)

with out_arcs.open("w") as g_arcs, \
     out_bary.open("w") as g_bary, \
     out_milieux.open("w") as g_milieux, \
     out_dual_polyedrique.open("w") as g_dual:

    for A, B, C in triangles:

        if visible_triangle(A, B, C):

            G = (A + B + C) / 3

            M_AB = (A + B) / 2
            M_BC = (B + C) / 2
            M_CA = (C + A) / 2

            # arêtes du maillage primal
            write_segment(g_arcs, A, B)
            write_segment(g_arcs, B, C)
            write_segment(g_arcs, C, A)

            # barycentre
            g_bary.write(f"{G[0]} {G[1]} {G[2]}\n")

            # milieux d'arêtes
            g_milieux.write(f"{M_AB[0]} {M_AB[1]} {M_AB[2]}\n")
            g_milieux.write(f"{M_BC[0]} {M_BC[1]} {M_BC[2]}\n")
            g_milieux.write(f"{M_CA[0]} {M_CA[1]} {M_CA[2]}\n")

            # maillage dual polyédrique
            write_segment(g_dual, G, M_AB)
            write_segment(g_dual, G, M_BC)
            write_segment(g_dual, G, M_CA)


with inp.open() as f, out_arcs.open("w") as g:
    next(f)  # ignore x y z

    pts = []

    for line in f:
        if line.strip():
            pts.append(np.array(list(map(float, line.split()))))

            if len(pts) == 3:
                A, B, C = pts

                if visible_triangle(A, B, C):
                    write_arc(g, A, B)
                    write_arc(g, B, C)
                    write_arc(g, C, A)

                pts = []