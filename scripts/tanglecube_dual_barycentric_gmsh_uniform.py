#!/usr/bin/env python3
"""
Maillage de la tanglecube algébrique de genre 5 :

    x^4 - 5 x^2 + y^4 - 5 y^2 + z^4 - 5 z^2 + c = 0,

avec c = 11.8 par défaut.

Étapes :
  1. échantillonnage de la fonction implicite sur une grille régulière ;
  2. extraction de l'isosurface triangulée par Flying Edges ;
  3. remaillage isotrope ACVD puis projection sur la surface algébrique ;
  4. import de la surface comme entité discrète Gmsh ;
  5. construction du dual barycentrique ;
  6. rendu PyVista : surface blanche, primal bleu, dual rouge.

Sorties :
  tanglecube_primal.msh
  tanglecube_dual_barycentric.msh
  tanglecube_dual_cells.csv
  tanglecube_primal_dual.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import gmsh


Point = Tuple[float, float, float]
Edge = Tuple[int, int]
Triangle = Tuple[int, int, int]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Maillage triangulaire de la tanglecube de genre 5 "
            "et construction de son dual barycentrique."
        )
    )
    parser.add_argument(
        "--constant",
        type=float,
        default=11.8,
        help="Constante c de l'équation algébrique (défaut : 11.8).",
    )
    parser.add_argument(
        "--extent",
        type=float,
        default=2.30,
        help="Demi-largeur de la boîte d'échantillonnage [-L,L]^3.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=180,
        help="Nombre de points de grille dans chaque direction.",
    )
    parser.add_argument(
        "--lc",
        type=float,
        default=0.12,
        help="Longueur cible approximative des arêtes du maillage primal.",
    )
    parser.add_argument(
        "--acvd-subdivisions",
        type=int,
        default=1,
        help=(
            "Nombre de subdivisions avant le remaillage ACVD. "
            "Utiliser 1 ou 2 ; 0 est souvent suffisant avec une grille fine."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Répertoire de sortie.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Ouvre l'interface Gmsh sur le dernier maillage généré.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Ne crée pas le rendu PyVista.",
    )
    parser.add_argument(
        "--off-screen",
        action="store_true",
        help="Crée uniquement le rendu, sans fenêtre interactive PyVista.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help=(
            "Chemin du rendu PNG/JPEG/TIFF/BMP ou PDF/SVG/EPS/PS "
            "(défaut : OUTDIR/tanglecube_primal_dual.png)."
        ),
    )
    parser.add_argument(
        "--primal-width",
        type=float,
        default=1.4,
        help="Épaisseur des arêtes primales bleues.",
    )
    parser.add_argument(
        "--dual-width",
        type=float,
        default=2.2,
        help="Épaisseur des arêtes duales rouges.",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=45.0,
        help="Azimut de la caméra en degrés (défaut : 45).",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=35.264,
        help="Élévation de la caméra en degrés (défaut : 35.264).",
    )
    return parser.parse_args()

def midpoint(p: Point, q: Point) -> Point:
    return (
        0.5 * (p[0] + q[0]),
        0.5 * (p[1] + q[1]),
        0.5 * (p[2] + q[2]),
    )


def barycenter(p: Point, q: Point, r: Point) -> Point:
    return (
        (p[0] + q[0] + r[0]) / 3.0,
        (p[1] + q[1] + r[1]) / 3.0,
        (p[2] + q[2] + r[2]) / 3.0,
    )


def canonical_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def tanglecube_field(x, y, z, constant: float):
    """Évalue le polynôme implicite de la tanglecube."""
    return (
        x**4 - 5.0 * x**2
        + y**4 - 5.0 * y**2
        + z**4 - 5.0 * z**2
        + constant
    )


def project_on_tanglecube(points, constant: float, iterations: int = 8):
    """Projette des points sur F=0 par itérations de Newton normales."""
    import numpy as np

    projected = np.asarray(points, dtype=float).copy()
    for _ in range(iterations):
        x = projected[:, 0]
        y = projected[:, 1]
        z = projected[:, 2]
        value = tanglecube_field(x, y, z, constant)
        gradient = np.column_stack(
            (
                4.0 * x**3 - 10.0 * x,
                4.0 * y**3 - 10.0 * y,
                4.0 * z**3 - 10.0 * z,
            )
        )
        norm2 = np.einsum("ij,ij->i", gradient, gradient)
        mask = norm2 > 1.0e-20
        projected[mask] -= (
            value[mask] / norm2[mask]
        )[:, None] * gradient[mask]
    return projected


def mesh_quality_statistics(surface):
    """Retourne des indicateurs simples de qualité du maillage triangulaire."""
    import numpy as np

    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)[:, 1:4]
    points = np.asarray(surface.points, dtype=float)
    triangles = points[faces]

    a = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    b = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
    c = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)

    def angle(opposite, side1, side2):
        denominator = 2.0 * side1 * side2
        cosine = np.divide(
            side1 * side1 + side2 * side2 - opposite * opposite,
            denominator,
            out=np.ones_like(opposite),
            where=denominator > 0.0,
        )
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    angles = np.column_stack(
        (
            angle(a, b, c),
            angle(b, c, a),
            angle(c, a, b),
        )
    )
    lengths = np.concatenate((a, b, c))

    return {
        "min_angle": float(np.min(angles)),
        "angle_p01": float(np.percentile(angles, 1.0)),
        "angle_p05": float(np.percentile(angles, 5.0)),
        "mean_edge": float(np.mean(lengths)),
        "edge_cv": float(np.std(lengths) / np.mean(lengths)),
    }


def create_implicit_surface(
    constant: float,
    extent: float,
    resolution: int,
    lc: float,
    acvd_subdivisions: int,
):
    """Extrait puis remaillage isotropiquement la tanglecube."""
    try:
        import numpy as np
        import pyvista as pv
        import pyacvd
    except ImportError as exc:
        raise RuntimeError(
            "Le remaillage uniforme nécessite numpy, pyvista et pyacvd : "
            "python3 -m pip install numpy pyvista pyacvd"
        ) from exc

    coordinates = np.linspace(-extent, extent, resolution, dtype=float)
    x, y, z = np.meshgrid(
        coordinates, coordinates, coordinates, indexing="ij"
    )
    field = tanglecube_field(x, y, z, constant)

    field_min = float(np.min(field))
    field_max = float(np.max(field))
    if not (field_min <= 0.0 <= field_max):
        raise RuntimeError(
            "Le niveau zéro n'est pas contenu dans la boîte : "
            f"min={field_min:.6g}, max={field_max:.6g}."
        )

    spacing_value = 2.0 * extent / (resolution - 1)
    grid = pv.ImageData(
        dimensions=(resolution, resolution, resolution),
        spacing=(spacing_value,) * 3,
        origin=(-extent, -extent, -extent),
    )
    grid.point_data["tanglecube"] = field.ravel(order="F")

    raw = grid.contour(
        isosurfaces=[0.0],
        scalars="tanglecube",
        method="flying_edges",
    )
    raw = raw.extract_surface().triangulate().clean()

    if raw.n_points == 0 or raw.n_cells == 0:
        raise RuntimeError("L'extraction de l'isosurface a produit un maillage vide.")

    boundary = raw.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    if boundary.n_cells != 0:
        raise RuntimeError(
            "La tanglecube coupe la boîte d'échantillonnage. "
            "Augmentez --extent."
        )

    # Pour une triangulation quasi équilatérale :
    # aire moyenne d'un triangle ~ sqrt(3) lc^2 / 4 et F ~ 2V.
    target_vertices = max(
        500,
        int(2.0 * float(raw.area) / (math.sqrt(3.0) * lc * lc)),
    )
    target_vertices = min(target_vertices, raw.n_points - 1)

    clustering = pyacvd.Clustering(raw)
    if acvd_subdivisions > 0:
        clustering.subdivide(acvd_subdivisions)
    clustering.cluster(target_vertices)
    surface = clustering.create_mesh().triangulate().clean()

    # ACVD travaille sur la surface polygonale extraite. La projection de Newton
    # replace ensuite les sommets sur la surface algébrique exacte F=0.
    surface.points = project_on_tanglecube(
        surface.points,
        constant,
        iterations=10,
    )
    surface = surface.clean().triangulate()
    surface = surface.compute_normals(
        point_normals=True,
        cell_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False,
    )

    statistics = mesh_quality_statistics(surface)
    print(
        "Qualité primal : "
        f"angle min={statistics['min_angle']:.2f}°, "
        f"P1={statistics['angle_p01']:.2f}°, "
        f"P5={statistics['angle_p05']:.2f}°, "
        f"arête moyenne={statistics['mean_edge']:.4f}, "
        f"CV(arêtes)={statistics['edge_cv']:.3f}"
    )

    return surface


def import_surface_into_gmsh(
    surface,
    output_file: Path,
) -> Tuple[Dict[int, Point], List[Triangle]]:
    """
    Importe la triangulation PyVista comme surface discrète Gmsh.
    """
    import numpy as np

    gmsh.model.add("tanglecube_primal")
    surface_tag = gmsh.model.addDiscreteEntity(2)

    xyz = np.asarray(surface.points, dtype=float)
    node_tags = list(range(1, xyz.shape[0] + 1))

    gmsh.model.mesh.addNodes(
        2,
        surface_tag,
        node_tags,
        xyz.ravel().tolist(),
    )

    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise RuntimeError("La surface contient des cellules non triangulaires.")

    connectivity = (faces[:, 1:4] + 1).ravel().tolist()
    element_tags = list(range(1, faces.shape[0] + 1))

    gmsh.model.mesh.addElementsByType(
        surface_tag,
        2,  # Triangle linéaire à trois nœuds.
        element_tags,
        connectivity,
    )

    physical_surface = gmsh.model.addPhysicalGroup(
        2,
        [surface_tag],
    )
    gmsh.model.setPhysicalName(
        2,
        physical_surface,
        "tanglecube_primal",
    )

    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(str(output_file))

    points: Dict[int, Point] = {
        tag: (
            float(xyz[tag - 1, 0]),
            float(xyz[tag - 1, 1]),
            float(xyz[tag - 1, 2]),
        )
        for tag in node_tags
    }

    triangles: List[Triangle] = [
        (
            int(row[1]) + 1,
            int(row[2]) + 1,
            int(row[3]) + 1,
        )
        for row in faces
    ]

    return points, triangles


def create_primal_mesh(
    constant: float,
    extent: float,
    resolution: int,
    lc: float,
    acvd_subdivisions: int,
    output_file: Path,
) -> Tuple[Dict[int, Point], List[Triangle]]:
    surface = create_implicit_surface(
        constant,
        extent,
        resolution,
        lc,
        acvd_subdivisions,
    )
    return import_surface_into_gmsh(surface, output_file)

def create_barycentric_dual(
    primal_points: Dict[int, Point],
    triangles: Sequence[Triangle],
    output_file: Path,
    cell_map_file: Path,
) -> None:
    gmsh.clear()
    gmsh.model.add("tanglecube_dual_barycentric")

    surface_tag = gmsh.model.addDiscreteEntity(2)

    dual_points: Dict[int, Point] = {}
    primal_to_dual_node: Dict[int, int] = {}
    edge_to_midpoint_node: Dict[Edge, int] = {}
    face_to_barycenter_node: Dict[int, int] = {}

    next_node_tag = 1

    for primal_tag in sorted(primal_points):
        primal_to_dual_node[primal_tag] = next_node_tag
        dual_points[next_node_tag] = primal_points[primal_tag]
        next_node_tag += 1

    edges = sorted(
        {
            canonical_edge(a, b)
            for a, b, c in triangles
            for a, b in ((a, b), (b, c), (c, a))
        }
    )

    for edge in edges:
        a, b = edge
        edge_to_midpoint_node[edge] = next_node_tag
        dual_points[next_node_tag] = midpoint(
            primal_points[a],
            primal_points[b],
        )
        next_node_tag += 1

    for face_index, (a, b, c) in enumerate(triangles):
        face_to_barycenter_node[face_index] = next_node_tag
        dual_points[next_node_tag] = barycenter(
            primal_points[a],
            primal_points[b],
            primal_points[c],
        )
        next_node_tag += 1

    node_tags = sorted(dual_points)
    coordinates: List[float] = []

    for tag in node_tags:
        coordinates.extend(dual_points[tag])

    gmsh.model.mesh.addNodes(
        2,
        surface_tag,
        node_tags,
        coordinates,
    )

    quad_connectivity: List[int] = []
    quad_element_tags: List[int] = []
    cell_rows: List[Tuple[int, int, int]] = []

    next_element_tag = 1

    for face_index, (a, b, c) in enumerate(triangles):
        g = face_to_barycenter_node[face_index]
        mab = edge_to_midpoint_node[canonical_edge(a, b)]
        mbc = edge_to_midpoint_node[canonical_edge(b, c)]
        mca = edge_to_midpoint_node[canonical_edge(c, a)]

        local_quads = (
            (a, [primal_to_dual_node[a], mab, g, mca]),
            (b, [primal_to_dual_node[b], mbc, g, mab]),
            (c, [primal_to_dual_node[c], mca, g, mbc]),
        )

        for primal_vertex, quad in local_quads:
            quad_element_tags.append(next_element_tag)
            quad_connectivity.extend(quad)
            cell_rows.append(
                (
                    primal_vertex,
                    next_element_tag,
                    face_index + 1,
                )
            )
            next_element_tag += 1

    gmsh.model.mesh.addElementsByType(
        surface_tag,
        3,  # Quadrilatère bilinéaire à quatre nœuds.
        quad_element_tags,
        quad_connectivity,
    )

    physical_surface = gmsh.model.addPhysicalGroup(
        2,
        [surface_tag],
    )
    gmsh.model.setPhysicalName(
        2,
        physical_surface,
        "dual_barycentric",
    )

    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(str(output_file))

    with cell_map_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "primal_vertex_tag",
                "dual_quadrangle_element_tag",
                "primal_face_index",
            ]
        )
        writer.writerows(cell_rows)


def render_primal_and_dual(
    primal_points: Dict[int, Point],
    triangles: Sequence[Triangle],
    extent: float,
    screenshot_file: Path,
    off_screen: bool,
    primal_width: float,
    dual_width: float,
    camera_azimuth: float,
    camera_elevation: float,
) -> None:
    try:
        import numpy as np
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "Le rendu nécessite numpy et pyvista : "
            "python3 -m pip install numpy pyvista"
        ) from exc

    sorted_tags = sorted(primal_points)
    tag_to_index = {
        tag: i for i, tag in enumerate(sorted_tags)
    }
    xyz = np.asarray(
        [primal_points[tag] for tag in sorted_tags],
        dtype=float,
    )

    faces = np.asarray(
        [
            item
            for triangle in triangles
            for item in (
                3,
                *(tag_to_index[tag] for tag in triangle),
            )
        ],
        dtype=np.int64,
    )

    surface = pv.PolyData(xyz, faces=faces)
    surface = surface.compute_normals(
        point_normals=True,
        cell_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False,
    )

    point_normals = surface.point_data["Normals"]
    cell_normals = surface.cell_data["Normals"]

    primal_edges = sorted(
        {
            canonical_edge(a, b)
            for a, b, c in triangles
            for a, b in ((a, b), (b, c), (c, a))
        }
    )

    offset = 1.0e-3 * extent

    primal_line_points = []
    primal_lines = []

    for a, b in primal_edges:
        ia = tag_to_index[a]
        ib = tag_to_index[b]
        start = len(primal_line_points)

        primal_line_points.extend(
            (
                xyz[ia] + offset * point_normals[ia],
                xyz[ib] + offset * point_normals[ib],
            )
        )
        primal_lines.extend((2, start, start + 1))

    primal_wire = pv.PolyData(
        np.asarray(primal_line_points),
        lines=np.asarray(primal_lines, dtype=np.int64),
    )

    dual_line_points = []
    dual_lines = []

    for face_index, (a, b, c) in enumerate(triangles):
        ia = tag_to_index[a]
        ib = tag_to_index[b]
        ic = tag_to_index[c]

        pa = xyz[ia]
        pb = xyz[ib]
        pc = xyz[ic]

        g = (pa + pb + pc) / 3.0
        ng = cell_normals[face_index]

        edge_data = (
            ((pa + pb) / 2.0, point_normals[ia] + point_normals[ib]),
            ((pb + pc) / 2.0, point_normals[ib] + point_normals[ic]),
            ((pc + pa) / 2.0, point_normals[ic] + point_normals[ia]),
        )

        for m, nm in edge_data:
            nm_norm = float(np.linalg.norm(nm))
            nm = nm / nm_norm if nm_norm != 0.0 else ng

            start = len(dual_line_points)
            dual_line_points.extend(
                (
                    g + 2.0 * offset * ng,
                    m + 2.0 * offset * nm,
                )
            )
            dual_lines.extend((2, start, start + 1))

    dual_wire = pv.PolyData(
        np.asarray(dual_line_points),
        lines=np.asarray(dual_lines, dtype=np.int64),
    )

    screenshot_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plotter = pv.Plotter(
        off_screen=off_screen,
        window_size=(1800, 1800),
    )
    plotter.set_background("white")

    plotter.add_mesh(
        surface,
        color="white",
        opacity=1.0,
        smooth_shading=True,
        show_edges=False,
        lighting=True,
    )
    plotter.add_mesh(
        primal_wire,
        color="blue",
        line_width=primal_width,
        render_lines_as_tubes=False,
        lighting=False,
    )
    plotter.add_mesh(
        dual_wire,
        color="red",
        line_width=dual_width,
        render_lines_as_tubes=False,
        lighting=False,
    )

    # Position sphérique explicite de la caméra.
    azimuth = math.radians(camera_azimuth)
    elevation = math.radians(camera_elevation)
    scene_scale = max(float(np.linalg.norm(point)) for point in xyz)
    camera_distance = 3.0 * scene_scale
    plotter.camera.focal_point = (0.0, 0.0, 0.0)
    plotter.camera.position = (
        camera_distance * math.cos(elevation) * math.cos(azimuth),
        camera_distance * math.cos(elevation) * math.sin(azimuth),
        camera_distance * math.sin(elevation),
    )
    plotter.camera.up = (0.0, 0.0, 1.0)

    # Recadrage serré conservant la direction de vue. zoom("tight") n'est pas
    # utilisé, car PyVista réoriente alors la caméra suivant le plan xy.
    view_direction = np.asarray(plotter.camera.focal_point) - np.asarray(
        plotter.camera.position
    )
    view_direction /= np.linalg.norm(view_direction)

    view_up = np.asarray(plotter.camera.up, dtype=float)
    camera_right = np.cross(view_direction, view_up)
    camera_right /= np.linalg.norm(camera_right)
    view_up = np.cross(camera_right, view_direction)
    view_up /= np.linalg.norm(view_up)
    plotter.camera.up = tuple(view_up)

    rendered_points = np.vstack(
        (
            xyz,
            np.asarray(primal_line_points),
            np.asarray(dual_line_points),
        )
    )
    relative_points = rendered_points - np.asarray(plotter.camera.focal_point)
    half_width = float(np.max(np.abs(relative_points @ camera_right)))
    half_height = float(np.max(np.abs(relative_points @ view_up)))

    # Adapte la page au rapport largeur/hauteur de la projection.
    projected_aspect_ratio = half_width / half_height
    maximum_dimension = 1800
    if projected_aspect_ratio >= 1.0:
        window_width = maximum_dimension
        window_height = max(1, round(maximum_dimension / projected_aspect_ratio))
    else:
        window_width = max(1, round(maximum_dimension * projected_aspect_ratio))
        window_height = maximum_dimension
    plotter.window_size = (window_width, window_height)

    window_width, window_height = plotter.window_size
    aspect_ratio = window_width / window_height
    plotter.camera.enable_parallel_projection()
    plotter.camera.parallel_scale = 1.03 * max(
        half_height, half_width / aspect_ratio
    )
    plotter.reset_camera_clipping_range()
    plotter.enable_anti_aliasing("ssaa")

    raster_extensions = {".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff"}
    graphic_extensions = {".pdf", ".svg", ".eps", ".ps", ".tex"}
    extension = screenshot_file.suffix.lower()

    if extension in graphic_extensions:
        plotter.show(auto_close=False)
        plotter.save_graphic(
            screenshot_file,
            title="Maillage primal et dual barycentrique de la tanglecube",
            raster=True,
        )
        plotter.close()
    elif extension in raster_extensions:
        plotter.show(
            screenshot=str(screenshot_file),
            auto_close=True,
            interactive=not off_screen,
        )
    else:
        plotter.close()
        supported = ", ".join(sorted(raster_extensions | graphic_extensions))
        raise ValueError(
            f"Extension de rendu non prise en charge : {extension or '(aucune)'}. "
            f"Extensions acceptées : {supported}"
        )


def check_parameters(
    constant: float,
    extent: float,
    resolution: int,
    lc: float,
    acvd_subdivisions: int,
) -> None:
    if not math.isfinite(constant):
        raise ValueError("--constant doit être fini.")

    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError("--extent doit être strictement positif.")

    if resolution < 60:
        raise ValueError(
            "--resolution doit être supérieur ou égal à 60."
        )

    if not math.isfinite(lc) or lc <= 0.0:
        raise ValueError("--lc doit être strictement positif.")

    if acvd_subdivisions < 0 or acvd_subdivisions > 3:
        raise ValueError("--acvd-subdivisions doit être compris entre 0 et 3.")

def main() -> int:
    args = parse_arguments()

    check_parameters(
        args.constant,
        args.extent,
        args.resolution,
        args.lc,
        args.acvd_subdivisions,
    )
    if not math.isfinite(args.camera_azimuth):
        raise ValueError("--camera-azimuth doit être un nombre fini.")
    if not math.isfinite(args.camera_elevation) or not (
        -89.0 < args.camera_elevation < 89.0
    ):
        raise ValueError(
            "--camera-elevation doit être comprise entre -89 et 89 degrés."
        )

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    primal_file = args.outdir / "tanglecube_primal.msh"
    dual_file = args.outdir / "tanglecube_dual_barycentric.msh"
    cell_map_file = args.outdir / "tanglecube_dual_cells.csv"
    screenshot_file = (
        args.screenshot
        or args.outdir / "tanglecube_primal_dual.png"
    )

    gmsh.initialize(sys.argv)

    try:
        gmsh.option.setNumber("General.Terminal", 1)

        primal_points, triangles = create_primal_mesh(
            args.constant,
            args.extent,
            args.resolution,
            args.lc,
            args.acvd_subdivisions,
            primal_file,
        )

        create_barycentric_dual(
            primal_points,
            triangles,
            dual_file,
            cell_map_file,
        )

        edges = {
            canonical_edge(a, b)
            for a, b, c in triangles
            for a, b in ((a, b), (b, c), (c, a))
        }
        vertices_count = len(primal_points)
        edges_count = len(edges)
        faces_count = len(triangles)
        euler_characteristic = (
            vertices_count - edges_count + faces_count
        )
        genus = (2 - euler_characteristic) // 2

        print(f"Maillage primal       : {primal_file}")
        print(f"Maillage dual         : {dual_file}")
        print(f"Correspondance        : {cell_map_file}")
        print(f"Sommets primaux       : {vertices_count}")
        print(f"Arêtes primales       : {edges_count}")
        print(f"Triangles             : {faces_count}")
        print(f"Caractéristique Euler : {euler_characteristic}")
        print(f"Genre calculé         : {genus}")
        print(f"Quadrilatères duaux   : {3 * faces_count}")

        if not args.no_render:
            render_primal_and_dual(
                primal_points,
                triangles,
                args.extent,
                screenshot_file,
                args.off_screen,
                args.primal_width,
                args.dual_width,
                args.camera_azimuth,
                args.camera_elevation,
            )
            print(f"Rendu                 : {screenshot_file}")

        if args.gui and "-nopopup" not in sys.argv:
            gmsh.fltk.run()

    finally:
        gmsh.finalize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python3 tanglecube_dual_barycentric_gmsh_uniform.py --constant 11.8 --extent 2.30 --resolution 180 --lc 0.15 --acvd-subdivisions 1 --outdir outputs

"""
python3 tanglecube_dual_barycentric_gmsh_uniform.py \
  --constant 11.8 \
  --extent 2.30 \
  --resolution 180 \
  --lc 0.4 \
  --primal-width 8 \
  --dual-width 8 \
  --acvd-subdivisions 1 \
  --camera-azimuth 120 \
  --camera-elevation 20 \
  --off-screen \
  --outdir outputs \
  --screenshot outputs/tanglecube_primal_dual.pdf
"""