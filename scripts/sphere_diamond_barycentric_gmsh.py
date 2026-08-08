#!/usr/bin/env python3
"""Construit les diamants plats DDFV d'un maillage triangulaire sphérique.

Pour chaque arête primale sigma = [A, B], les deux triangles adjacents ont
pour barycentres projetés radialement sur la sphère G et H. Le segment [G, H]
est l'arête duale. Le plan du diamant est le plan affine engendré par les
directions B-A et H-G et passant par (A+B+G+H)/4. Les quatre sommets du
diamant sont les projections orthogonales de A, G, B et H sur ce plan.

Sorties par défaut :
  sphere_primal.msh
  sphere_diamonds.msh
  sphere_diamonds.csv
  sphere_diamonds.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import gmsh

from sphere_dual_barycentric_gmsh import (
    Edge,
    Point,
    Triangle,
    canonical_edge,
    create_primal_mesh,
)


Diamond = Tuple[Point, Point, Point, Point]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Maillage d'une sphère et diamants plats DDFV."
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--lc", type=float, default=0.18)
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-render", action="store_true")
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
            "(défaut : OUTDIR/sphere_diamonds.png)."
        ),
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


def add(p: Point, q: Point) -> Point:
    return (p[0] + q[0], p[1] + q[1], p[2] + q[2])


def subtract(p: Point, q: Point) -> Point:
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def scale(a: float, p: Point) -> Point:
    return (a * p[0], a * p[1], a * p[2])


def dot(p: Point, q: Point) -> float:
    return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]


def cross(p: Point, q: Point) -> Point:
    return (
        p[1] * q[2] - p[2] * q[1],
        p[2] * q[0] - p[0] * q[2],
        p[0] * q[1] - p[1] * q[0],
    )


def norm(p: Point) -> float:
    return math.sqrt(dot(p, p))


def projected_barycenter_on_sphere(
    triangle: Triangle, points: Dict[int, Point], radius: float
) -> Point:
    a, b, c = (points[tag] for tag in triangle)
    g = scale(1.0 / 3.0, add(add(a, b), c))
    length = norm(g)
    if length <= 1.0e-15 * radius:
        raise RuntimeError("Barycentre trop proche du centre de la sphère.")
    return scale(radius / length, g)


def project_on_plane(p: Point, origin: Point, unit_normal: Point) -> Point:
    """Projection dans la direction sigma x dual_sigma (donc orthogonale)."""
    return subtract(p, scale(dot(subtract(p, origin), unit_normal), unit_normal))


def build_diamonds(
    points: Dict[int, Point], triangles: Sequence[Triangle], radius: float
) -> Tuple[List[Tuple[Edge, Tuple[int, int], Diamond]], List[Point]]:
    """Retourne un diamant par arête intérieure et les centres de faces lisses."""
    face_centers = [
        projected_barycenter_on_sphere(triangle, points, radius)
        for triangle in triangles
    ]
    adjacent_faces: Dict[Edge, List[int]] = defaultdict(list)
    for face_index, (a, b, c) in enumerate(triangles):
        for edge in ((a, b), (b, c), (c, a)):
            adjacent_faces[canonical_edge(*edge)].append(face_index)

    diamonds: List[Tuple[Edge, Tuple[int, int], Diamond]] = []
    tolerance = 1.0e-13 * radius * radius
    for edge in sorted(adjacent_faces):
        faces = adjacent_faces[edge]
        if len(faces) != 2:
            raise RuntimeError(
                f"L'arête {edge} possède {len(faces)} faces adjacentes au lieu de 2."
            )

        a, b = (points[tag] for tag in edge)
        g, h = (face_centers[index] for index in faces)
        primal_direction = subtract(b, a)
        dual_direction = subtract(h, g)
        normal = cross(primal_direction, dual_direction)
        normal_length = norm(normal)
        if normal_length <= tolerance:
            raise RuntimeError(f"Plan du diamant dégénéré pour l'arête {edge}.")
        unit_normal = scale(1.0 / normal_length, normal)

        # Centre commun aux quatre milieux décrits dans la section Diamond mesh.
        plane_origin = scale(0.25, add(add(a, b), add(g, h)))
        projected = tuple(
            project_on_plane(p, plane_origin, unit_normal) for p in (a, g, b, h)
        )
        diamonds.append((edge, (faces[0], faces[1]), projected))

    return diamonds, face_centers


def write_diamond_mesh(
    diamonds: Sequence[Tuple[Edge, Tuple[int, int], Diamond]],
    output_file: Path,
    csv_file: Path,
) -> None:
    gmsh.clear()
    gmsh.model.add("sphere_diamonds")
    surface_tag = gmsh.model.addDiscreteEntity(2)

    node_tags: List[int] = []
    coordinates: List[float] = []
    element_tags: List[int] = []
    connectivity: List[int] = []
    rows = []

    for element_tag, (edge, faces, vertices) in enumerate(diamonds, start=1):
        local_nodes = []
        for vertex in vertices:
            node_tag = len(node_tags) + 1
            node_tags.append(node_tag)
            coordinates.extend(vertex)
            local_nodes.append(node_tag)
        element_tags.append(element_tag)
        connectivity.extend(local_nodes)
        rows.append((element_tag, edge[0], edge[1], faces[0] + 1, faces[1] + 1))

    gmsh.model.mesh.addNodes(2, surface_tag, node_tags, coordinates)
    gmsh.model.mesh.addElementsByType(
        surface_tag, 3, element_tags, connectivity  # Gmsh type 3: quad à 4 nœuds
    )
    physical_tag = gmsh.model.addPhysicalGroup(2, [surface_tag])
    gmsh.model.setPhysicalName(2, physical_tag, "flat_diamonds")
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(str(output_file))

    with csv_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["diamond_tag", "primal_a", "primal_b", "face_k", "face_l"]
        )
        writer.writerows(rows)


def render(
    points: Dict[int, Point],
    triangles: Sequence[Triangle],
    diamonds: Sequence[Tuple[Edge, Tuple[int, int], Diamond]],
    screenshot: Path,
    off_screen: bool,
    camera_azimuth: float,
    camera_elevation: float,
) -> None:
    try:
        import numpy as np
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("Le rendu nécessite numpy et pyvista.") from exc

    tags = sorted(points)
    indices = {tag: index for index, tag in enumerate(tags)}
    xyz = np.asarray([points[tag] for tag in tags])
    faces = np.asarray(
        [value for tri in triangles for value in (3, *(indices[t] for t in tri))],
        dtype=np.int64,
    )
    sphere = pv.PolyData(xyz, faces=faces)

    diamond_points = []
    diamond_faces = []
    for _, _, vertices in diamonds:
        first = len(diamond_points)
        diamond_points.extend(vertices)
        diamond_faces.extend((4, first, first + 1, first + 2, first + 3))
    diamond_mesh = pv.PolyData(
        np.asarray(diamond_points), faces=np.asarray(diamond_faces, dtype=np.int64)
    )

    screenshot.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=off_screen, window_size=(1600, 1600))
    plotter.set_background("white")
    plotter.add_mesh(sphere, color="white", opacity=0.35, show_edges=True)
    plotter.add_mesh(
        diamond_mesh,
        color="green",
        opacity=1,
        show_edges=True,
        edge_color="limegreen",
        line_width=12,
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

    # Recadrage serré qui conserve la direction de vue. zoom("tight") n'est
    # pas utilisé, car PyVista réoriente alors la caméra suivant le plan xy.
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

    rendered_points = np.vstack((xyz, np.asarray(diamond_points)))
    relative_points = rendered_points - np.asarray(plotter.camera.focal_point)
    half_width = float(np.max(np.abs(relative_points @ camera_right)))
    half_height = float(np.max(np.abs(relative_points @ view_up)))

    # Adapte la page au rapport largeur/hauteur de la projection.
    projected_aspect_ratio = half_width / half_height
    maximum_dimension = 1600
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
    extension = screenshot.suffix.lower()

    if extension in graphic_extensions:
        plotter.show(auto_close=False)
        plotter.save_graphic(
            screenshot,
            title="Diamants barycentriques de la sphère",
            raster=True,
        )
        plotter.close()
    elif extension in raster_extensions:
        plotter.show(screenshot=str(screenshot), auto_close=True)
    else:
        plotter.close()
        supported = ", ".join(sorted(raster_extensions | graphic_extensions))
        raise ValueError(
            f"Extension de rendu non prise en charge : {extension or '(aucune)'}. "
            f"Extensions acceptées : {supported}"
        )


def main() -> int:
    args = parse_arguments()
    if not math.isfinite(args.radius) or args.radius <= 0.0:
        raise ValueError("--radius doit être strictement positif.")
    if not math.isfinite(args.lc) or args.lc <= 0.0:
        raise ValueError("--lc doit être strictement positif.")
    if not math.isfinite(args.camera_azimuth):
        raise ValueError("--camera-azimuth doit être un nombre fini.")
    if not math.isfinite(args.camera_elevation) or not (
        -89.0 < args.camera_elevation < 89.0
    ):
        raise ValueError(
            "--camera-elevation doit être comprise entre -89 et 89 degrés."
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    primal_file = args.outdir / "sphere_primal.msh"
    diamond_file = args.outdir / "sphere_diamonds.msh"
    csv_file = args.outdir / "sphere_diamonds.csv"
    screenshot = args.screenshot or args.outdir / "sphere_diamonds.png"

    # Les options propres à ce script ont déjà été traitées par argparse.
    gmsh.initialize([])
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        points, triangles = create_primal_mesh(args.radius, args.lc, primal_file)
        diamonds, _ = build_diamonds(points, triangles, args.radius)
        write_diamond_mesh(diamonds, diamond_file, csv_file)

        print(f"Maillage primal : {primal_file}")
        print(f"Diamants         : {diamond_file}")
        print(f"Correspondance   : {csv_file}")
        print(f"Nombre de diamants : {len(diamonds)}")

        if not args.no_render:
            render(
                points,
                triangles,
                diamonds,
                screenshot,
                args.off_screen,
                args.camera_azimuth,
                args.camera_elevation,
            )
            print(f"Rendu            : {screenshot}")
        if args.gui and "-nopopup" not in sys.argv:
            gmsh.fltk.run()
    finally:
        gmsh.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python3 sphere_diamond_barycentric_gmsh.py --radius 1 --lc 0.5 --outdir outputs

"""
python3 sphere_diamond_barycentric_gmsh.py   --radius 1   --lc 1.2   --camera-azimuth 115   --camera-elevation 15   --off-screen   --outdir outputs   --screenshot outputs/sphere_diamonds.pdf
"""