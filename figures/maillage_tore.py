from pathlib import Path
import math

import numpy as np


HERE = Path(__file__).resolve().parent
inp = HERE.parent / "data" / "faces3D_tore_complet.dat"

out_arcs = HERE / "maillage_tore_arcs_avant.dat"
out_primal_polyedrique = HERE / "maillage_tore_primal_polyedrique_avant.dat"
out_bary = HERE / "maillage_tore_barycentres_avant.dat"
out_milieux = HERE / "maillage_tore_milieux_avant.dat"
out_dual_polyedrique = HERE / "maillage_tore_dual_polyedrique_avant.dat"
out_dual_patches = HERE / "maillage_tore_faces_duales_polyedriques.dat"
out_common_refinement = (
    HERE / "maillage_tore_raffinement_primal_dual_polyedrique.dat"
)
DUAL_PATCH_CHUNK_SIZE = 50
COMMON_REFINEMENT_CHUNK_SIZE = 100

R = 1.0
r = 0.3
N_ARC = 10
N_VISIBILITY = 24

# Direction de l'observateur correspondant à view={70}{22} dans PGFPlots.
az = math.radians(-10)
el = math.radians(22)
view_dir = np.array([
    math.cos(az) * math.cos(el),
    math.sin(az) * math.cos(el),
    math.sin(el),
])


def read_triangles(filename):
    triangles = []
    with filename.open() as f:
        next(f)  # ignore l'en-tête x y z
        points = []
        for line in f:
            if line.strip():
                points.append(np.array(list(map(float, line.split()))))
                if len(points) == 3:
                    triangles.append(tuple(points))
                    points = []
    return triangles


def write_polyline(output, points):
    for point in points:
        output.write(f"{point[0]} {point[1]} {point[2]}\n")
    output.write("\n")


def torus_parameters(point):
    x, y, z = point
    u = math.atan2(y, x)
    rho = math.hypot(x, y)
    v = math.atan2(z, rho - R)
    return u, v


def unwrap_angle(angle, reference):
    while angle - reference > math.pi:
        angle -= 2 * math.pi
    while angle - reference < -math.pi:
        angle += 2 * math.pi
    return angle


def torus_point(u, v):
    return np.array([
        (R + r * math.cos(v)) * math.cos(u),
        (R + r * math.cos(v)) * math.sin(u),
        r * math.sin(v),
    ])


def visible_torus_point(point):
    """Critère historique, réservé aux arêtes courbes du maillage primal."""
    x, y, _ = point
    rho = math.hypot(x, y)
    circle_center = np.array([R * x / rho, R * y / rho, 0.0])
    normal = point - circle_center
    return np.dot(normal, view_dir) > 0


def arc_points_torus(A, B, n=N_ARC):
    uA, vA = torus_parameters(A)
    uB, vB = torus_parameters(B)
    uB = unwrap_angle(uB, uA)
    vB = unwrap_angle(vB, vA)

    for t in np.linspace(0, 1, n):
        yield torus_point((1 - t) * uA + t * uB,
                          (1 - t) * vA + t * vB)


triangles = read_triangles(inp)
triangle_array = np.asarray(triangles)

# Raffinement commun primal-dual : chaque triangle ABC est partagé en trois
# quadrilatères dont les côtés sont uniquement des arêtes primales ou duales.
with out_common_refinement.open("w") as output:
    output.write("x y z\n")
    for A, B, C in triangles:
        G = (A + B + C) / 3
        M_AB = (A + B) / 2
        M_BC = (B + C) / 2
        M_CA = (C + A) / 2
        for quadrilateral in (
            (A, M_AB, G, M_CA),
            (B, M_BC, G, M_AB),
            (C, M_CA, G, M_BC),
        ):
            for point in quadrilateral:
                output.write(f"{point[0]} {point[1]} {point[2]}\n")

for chunk_index, start in enumerate(
        range(0, len(triangles), COMMON_REFINEMENT_CHUNK_SIZE)):
    chunk_path = HERE / (
        f"maillage_tore_raffinement_primal_dual_polyedrique_"
        f"{chunk_index:02d}.dat"
    )
    with chunk_path.open("w") as output:
        output.write("x y z\n")
        for A, B, C in triangles[
                start:start + COMMON_REFINEMENT_CHUNK_SIZE]:
            G = (A + B + C) / 3
            M_AB = (A + B) / 2
            M_BC = (B + C) / 2
            M_CA = (C + A) / 2
            for quadrilateral in (
                (A, M_AB, G, M_CA),
                (B, M_BC, G, M_AB),
                (C, M_CA, G, M_BC),
            ):
                for point in quadrilateral:
                    output.write(f"{point[0]} {point[1]} {point[2]}\n")

# Quantités constantes du test rayon-triangle de Möller--Trumbore.
ray_origins = triangle_array[:, 0]
edge_1 = triangle_array[:, 1] - ray_origins
edge_2 = triangle_array[:, 2] - ray_origins
ray_cross_edge_2 = np.cross(view_dir, edge_2)
ray_determinants = np.einsum("ij,ij->i", edge_1, ray_cross_edge_2)


def visible_point(point):
    """Vrai si aucun triangle du tore n'occulte point vers l'observateur."""
    epsilon = 1e-9
    valid = np.abs(ray_determinants) > epsilon
    inverse_determinants = np.zeros_like(ray_determinants)
    inverse_determinants[valid] = 1.0 / ray_determinants[valid]

    origins_to_point = point - ray_origins
    barycentric_u = (
        np.einsum("ij,ij->i", origins_to_point, ray_cross_edge_2)
        * inverse_determinants
    )
    valid &= (barycentric_u >= -epsilon) & (barycentric_u <= 1 + epsilon)

    cross_q = np.cross(origins_to_point, edge_1)
    barycentric_v = (
        np.einsum("j,ij->i", view_dir, cross_q)
        * inverse_determinants
    )
    valid &= (barycentric_v >= -epsilon)
    valid &= (barycentric_u + barycentric_v <= 1 + epsilon)

    distance = (
        np.einsum("ij,ij->i", edge_2, cross_q)
        * inverse_determinants
    )

    # Les intersections à distance nulle sont le triangle qui porte le
    # segment (ou son voisin sur une arête), et ne sont pas des occultations.
    return not np.any(valid & (distance > 1e-7))


def transition_point(hidden, visible, iterations=20):
    """Approche la frontière visible/cachée par dichotomie."""
    for _ in range(iterations):
        middle = (hidden + visible) / 2
        if visible_point(middle):
            visible = middle
        else:
            hidden = middle
    return visible


def visible_parts(A, B, samples=N_VISIBILITY):
    """Découpe AB en polylignes visibles, y compris aux silhouettes."""
    points = [(1 - t) * A + t * B for t in np.linspace(0, 1, samples + 1)]
    states = [visible_point(point) for point in points]
    current = [points[0]] if states[0] else []

    for previous, point, was_visible, is_visible in zip(
            points, points[1:], states, states[1:]):
        if was_visible and is_visible:
            current.append(point)
        elif was_visible and not is_visible:
            current.append(transition_point(point, previous))
            yield current
            current = []
        elif not was_visible and is_visible:
            current = [transition_point(previous, point), point]

    if current:
        yield current


def torus_normal(point):
    x, y, _ = point
    rho = math.hypot(x, y)
    circle_center = np.array([R * x / rho, R * y / rho, 0.0])
    normal = point - circle_center
    return normal / np.linalg.norm(normal)


def ordered_dual_cell(vertex, incident_triangles):
    """Sommets alternés milieu/barycentre de la cellule duale de vertex."""
    cell_points = {}
    for triangle in incident_triangles:
        others = [point for point in triangle if tuple(point) != tuple(vertex)]
        barycenter = sum(triangle) / 3
        cell_points[tuple(barycenter)] = barycenter
        for other in others:
            midpoint = (vertex + other) / 2
            cell_points[tuple(midpoint)] = midpoint

    normal = torus_normal(vertex)
    radial = np.array([vertex[0], vertex[1], 0.0])
    tangent_1 = np.cross(normal, radial)
    if np.linalg.norm(tangent_1) < 1e-12:
        tangent_1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(normal, tangent_1)

    def angle(point):
        direction = point - vertex
        return math.atan2(np.dot(direction, tangent_2),
                          np.dot(direction, tangent_1))

    return sorted(cell_points.values(), key=angle)


written_primal_edges = set()

with (
    out_arcs.open("w") as g_arcs,
    out_bary.open("w") as g_bary,
    out_milieux.open("w") as g_milieux,
    out_dual_polyedrique.open("w") as g_dual,
):
    for A, B, C in triangles:
        G = (A + B + C) / 3
        midpoints = ((A + B) / 2, (B + C) / 2, (C + A) / 2)

        # On conserve le rendu historique des arêtes primales : la correction
        # d'occlusion détaillée concerne ici le maillage dual polyédrique.
        if all(visible_torus_point(point) for point in (A, B, C)):
            for P, Q in ((A, B), (B, C), (C, A)):
                edge_key = tuple(sorted((tuple(P), tuple(Q))))
                if edge_key not in written_primal_edges:
                    write_polyline(g_arcs, arc_points_torus(P, Q))
                    written_primal_edges.add(edge_key)

        if visible_point(G):
            g_bary.write(f"{G[0]} {G[1]} {G[2]}\n")

        for midpoint in midpoints:
            if visible_point(midpoint):
                g_milieux.write(
                    f"{midpoint[0]} {midpoint[1]} {midpoint[2]}\n"
                )
            for part in visible_parts(G, midpoint):
                # Une demi-arête duale est rectiligne : les points de
                # sondage intermédiaires ne sont pas utiles à PGFPlots.
                write_polyline(g_dual, (part[0], part[-1]))


# Arêtes droites du maillage primal, découpées par la même occultation exacte
# que le dual. Elles seront retracées après les patchs duaux dans PGFPlots.
written_primal_edges.clear()
with out_primal_polyedrique.open("w") as output:
    for triangle in triangles:
        A, B, C = triangle
        for P, Q in ((A, B), (B, C), (C, A)):
            edge_key = tuple(sorted((tuple(P), tuple(Q))))
            if edge_key in written_primal_edges:
                continue
            written_primal_edges.add(edge_key)
            for part in visible_parts(P, Q):
                write_polyline(output, (part[0], part[-1]))


# Construction des cellules duales complètes. Une cellule associée à un sommet
# de valence k alterne k milieux d'arêtes et k barycentres. PGFPlots demande un
# nombre de sommets constant : on complète à 16 (valence maximale 8) en
# répétant le dernier point, ce qui ne crée aucune arête visible supplémentaire.
incident_by_vertex = {}
for triangle in triangles:
    for vertex in triangle:
        incident_by_vertex.setdefault(tuple(vertex), []).append(triangle)

dual_cells = [
    ordered_dual_cell(np.array(vertex), incident)
    for vertex, incident in incident_by_vertex.items()
]
# PGFPlots effectue sinon ce tri en mémoire et dépasse facilement la capacité
# de pdfLaTeX. view_dir pointe vers l'observateur : profondeur croissante =
# cellules de plus en plus proches, donc ordre naturel du peintre.
dual_cells.sort(
    key=lambda cell: np.dot(np.mean(cell, axis=0), view_dir)
)
max_cell_size = max(map(len, dual_cells))

with out_dual_patches.open("w") as output:
    output.write("x y z\n")
    for cell in dual_cells:
        padded_cell = cell + [cell[-1]] * (max_cell_size - len(cell))
        for point in padded_cell:
            output.write(f"{point[0]} {point[1]} {point[2]}\n")

# Le type polygon de PGFPlots mémorise tout un \addplot. Les blocs restent dans
# l'ordre global arrière -> avant, mais limitent la mémoire de chaque tracé.
for chunk_index, start in enumerate(
        range(0, len(dual_cells), DUAL_PATCH_CHUNK_SIZE)):
    chunk_path = HERE / (
        f"maillage_tore_faces_duales_polyedriques_{chunk_index:02d}.dat"
    )
    with chunk_path.open("w") as output:
        output.write("x y z\n")
        for cell in dual_cells[start:start + DUAL_PATCH_CHUNK_SIZE]:
            padded_cell = cell + [cell[-1]] * (max_cell_size - len(cell))
            for point in padded_cell:
                output.write(f"{point[0]} {point[1]} {point[2]}\n")
