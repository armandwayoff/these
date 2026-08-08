#!/usr/bin/env python3
"""
Génère avec Gmsh :
  1. un maillage simplicial (triangulaire) d'un tore ;
  2. son maillage dual barycentrique ;
  3. un rendu PyVista superposant primal et dual.

Le dual barycentrique est construit sur la surface polyédrique triangulée :
pour chaque triangle (a,b,c), on crée trois quadrilatères
  [a, m_ab, g, m_ca],
  [b, m_bc, g, m_ab],
  [c, m_ca, g, m_bc],
où m_ij est le milieu de l'arête (i,j) et g le barycentre du triangle.

Sorties :
  torus_primal.msh
  torus_dual_barycentric.msh
  torus_dual_cells.csv
  torus_primal_dual.png
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
        description="Maillage triangulaire d'un tore et dual barycentrique."
    )
    parser.add_argument(
        "--major-radius",
        type=float,
        default=1.5,
        help="Rayon majeur du tore.",
    )
    parser.add_argument(
        "--minor-radius",
        type=float,
        default=0.5,
        help="Rayon mineur du tore.",
    )
    parser.add_argument(
        "--lc",
        type=float,
        default=0.16,
        help="Taille caractéristique du maillage primal.",
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
            "(défaut : OUTDIR/torus_primal_dual.png)."
        ),
    )
    parser.add_argument(
        "--primal-width",
        type=float,
        default=1.8,
        help="Épaisseur des arêtes primales bleues.",
    )
    parser.add_argument(
        "--dual-width",
        type=float,
        default=2.8,
        help="Épaisseur des arêtes duales rouges.",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=-45.0,
        help="Azimut de la caméra en degrés (défaut : -45).",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=29.3,
        help="Élévation de la caméra en degrés (défaut : 29.3).",
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


def extract_linear_triangles() -> Tuple[Dict[int, Point], List[Triangle]]:
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()

    points: Dict[int, Point] = {}
    for i, tag in enumerate(node_tags):
        points[int(tag)] = (
            float(coordinates[3 * i]),
            float(coordinates[3 * i + 1]),
            float(coordinates[3 * i + 2]),
        )

    triangles: List[Triangle] = []

    for _, surface_tag in gmsh.model.getEntities(2):
        element_types, _, element_node_tags = gmsh.model.mesh.getElements(
            2, surface_tag
        )

        for element_type, connectivity in zip(
            element_types, element_node_tags
        ):
            properties = gmsh.model.mesh.getElementProperties(element_type)
            element_name = properties[0]
            number_of_nodes = properties[3]

            if element_name != "Triangle 3" or number_of_nodes != 3:
                continue

            for i in range(0, len(connectivity), 3):
                triangles.append(
                    (
                        int(connectivity[i]),
                        int(connectivity[i + 1]),
                        int(connectivity[i + 2]),
                    )
                )

    if not triangles:
        raise RuntimeError("Aucun triangle linéaire n'a été trouvé.")

    used_tags = {tag for triangle in triangles for tag in triangle}
    points = {tag: points[tag] for tag in used_tags}

    return points, triangles


def create_primal_mesh(
    major_radius: float,
    minor_radius: float,
    lc: float,
    output_file: Path,
) -> Tuple[Dict[int, Point], List[Triangle]]:
    gmsh.model.add("torus_primal")

    volume_tag = gmsh.model.occ.addTorus(
        0.0,
        0.0,
        0.0,
        major_radius,
        minor_radius,
    )
    gmsh.model.occ.synchronize()

    boundary_surfaces = gmsh.model.getBoundary(
        [(3, volume_tag)],
        oriented=False,
        recursive=False,
    )
    surface_tags = [
        tag for dim, tag in boundary_surfaces if dim == 2
    ]

    if not surface_tags:
        raise RuntimeError("La frontière surfacique du tore est vide.")

    physical_surface = gmsh.model.addPhysicalGroup(2, surface_tags)
    gmsh.model.setPhysicalName(
        2,
        physical_surface,
        "torus_primal",
    )

    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.removeDuplicateNodes()

    points, triangles = extract_linear_triangles()
    gmsh.write(str(output_file))

    return points, triangles


def create_barycentric_dual(
    primal_points: Dict[int, Point],
    triangles: Sequence[Triangle],
    output_file: Path,
    cell_map_file: Path,
) -> None:
    gmsh.clear()
    gmsh.model.add("torus_dual_barycentric")

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
            (
                a,
                [primal_to_dual_node[a], mab, g, mca],
            ),
            (
                b,
                [primal_to_dual_node[b], mbc, g, mab],
            ),
            (
                c,
                [primal_to_dual_node[c], mca, g, mbc],
            ),
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
        3,
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


def torus_normal(
    point,
    major_radius: float,
):
    """
    Normale extérieure au tore paramétrique centré sur l'axe Oz.
    """
    import numpy as np

    p = np.asarray(point, dtype=float)
    rho = math.hypot(float(p[0]), float(p[1]))

    if rho == 0.0:
        return np.asarray((0.0, 0.0, 1.0))

    centerline = np.asarray(
        (
            major_radius * p[0] / rho,
            major_radius * p[1] / rho,
            0.0,
        )
    )

    normal = p - centerline
    norm = float(np.linalg.norm(normal))

    if norm == 0.0:
        return np.asarray((0.0, 0.0, 1.0))

    return normal / norm


def render_primal_and_dual(
    primal_points: Dict[int, Point],
    triangles: Sequence[Triangle],
    major_radius: float,
    minor_radius: float,
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
            "Le rendu nécessite numpy et pyvista. "
            "Installez-les avec : "
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

    primal_edges = sorted(
        {
            canonical_edge(a, b)
            for a, b, c in triangles
            for a, b in ((a, b), (b, c), (c, a))
        }
    )

    offset = 2.0e-3 * minor_radius

    def shifted(point, amount: float):
        p = np.asarray(point, dtype=float)
        return p + amount * torus_normal(p, major_radius)

    primal_line_points = []
    primal_lines = []

    for a, b in primal_edges:
        start = len(primal_line_points)

        primal_line_points.extend(
            (
                shifted(primal_points[a], offset),
                shifted(primal_points[b], offset),
            )
        )
        primal_lines.extend((2, start, start + 1))

    primal_wire = pv.PolyData(
        np.asarray(primal_line_points),
        lines=np.asarray(primal_lines, dtype=np.int64),
    )

    dual_line_points = []
    dual_lines = []

    for a, b, c in triangles:
        pa = np.asarray(primal_points[a], dtype=float)
        pb = np.asarray(primal_points[b], dtype=float)
        pc = np.asarray(primal_points[c], dtype=float)

        g = (pa + pb + pc) / 3.0

        for m in (
            (pa + pb) / 2.0,
            (pb + pc) / 2.0,
            (pc + pa) / 2.0,
        ):
            start = len(dual_line_points)

            dual_line_points.extend(
                (
                    shifted(g, 2.0 * offset),
                    shifted(m, 2.0 * offset),
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
        window_size=(1600, 1600),
    )
    plotter.set_background("white")

    plotter.add_mesh(
        surface,
        color="white",
        opacity=1.0,
        smooth_shading=False,
        show_edges=False,
        lighting=True,
    )
    plotter.add_mesh(
        primal_wire,
        color="blue",
        line_width=primal_width,
        render_lines_as_tubes=True,
        lighting=False,
    )
    plotter.add_mesh(
        dual_wire,
        color="red",
        line_width=dual_width,
        render_lines_as_tubes=True,
        lighting=False,
    )

    # Position sphérique explicite de la caméra. Les valeurs par défaut
    # reproduisent approximativement l'ancienne position (3.4, -3.4, 2.7).
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

    # Adapte aussi la forme de la fenêtre à la projection du tore. Une vue
    # basse produit ainsi un PDF rectangulaire au lieu d'une page carrée avec
    # de grandes marges blanches en haut et en bas.
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
    # Marge de 3 % autour de la projection du tore.
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
            title="Maillage primal et dual barycentrique du tore",
            raster=True,
        )
        plotter.close()
    elif extension in raster_extensions:
        plotter.show(
            screenshot=str(screenshot_file),
            auto_close=True,
        )
    else:
        plotter.close()
        supported = ", ".join(sorted(raster_extensions | graphic_extensions))
        raise ValueError(
            f"Extension de rendu non prise en charge : {extension or '(aucune)'}. "
            f"Extensions acceptées : {supported}"
        )


def check_parameters(
    major_radius: float,
    minor_radius: float,
    lc: float,
) -> None:
    if not math.isfinite(major_radius) or major_radius <= 0.0:
        raise ValueError(
            "--major-radius doit être strictement positif."
        )

    if not math.isfinite(minor_radius) or minor_radius <= 0.0:
        raise ValueError(
            "--minor-radius doit être strictement positif."
        )

    if major_radius <= minor_radius:
        raise ValueError(
            "On impose --major-radius > --minor-radius "
            "pour obtenir un tore annulaire non auto-intersectant."
        )

    if not math.isfinite(lc) or lc <= 0.0:
        raise ValueError("--lc doit être strictement positif.")


def main() -> int:
    args = parse_arguments()

    check_parameters(
        args.major_radius,
        args.minor_radius,
        args.lc,
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

    primal_file = args.outdir / "torus_primal.msh"
    dual_file = (
        args.outdir / "torus_dual_barycentric.msh"
    )
    cell_map_file = args.outdir / "torus_dual_cells.csv"
    screenshot_file = (
        args.screenshot
        or args.outdir / "torus_primal_dual.png"
    )

    gmsh.initialize(sys.argv)

    try:
        gmsh.option.setNumber("General.Terminal", 1)

        primal_points, triangles = create_primal_mesh(
            args.major_radius,
            args.minor_radius,
            args.lc,
            primal_file,
        )

        create_barycentric_dual(
            primal_points,
            triangles,
            dual_file,
            cell_map_file,
        )

        print(f"Maillage primal       : {primal_file}")
        print(f"Maillage dual         : {dual_file}")
        print(f"Correspondance        : {cell_map_file}")
        print(f"Sommets primaux       : {len(primal_points)}")
        print(f"Triangles             : {len(triangles)}")
        print(
            f"Quadrilatères duaux   : {3 * len(triangles)}"
        )

        if not args.no_render:
            render_primal_and_dual(
                primal_points,
                triangles,
                args.major_radius,
                args.minor_radius,
                screenshot_file,
                args.off_screen,
                args.primal_width,
                args.dual_width,
                args.camera_azimuth,
                args.camera_elevation,
            )
            print(f"Rendu                  : {screenshot_file}")

        if args.gui and "-nopopup" not in sys.argv:
            gmsh.fltk.run()

    finally:
        gmsh.finalize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python3 torus_dual_barycentric_gmsh.py --major-radius 1.5 --minor-radius 0.5 --lc 0.16 --primal-width 4 --dual-width 4 --outdir outputs

"""
python3 torus_dual_barycentric_gmsh.py \
  --major-radius 1.5 \
  --minor-radius 0.5 \
  --lc 0.4 \
  --primal-width 8 \
  --dual-width 8 \
  --camera-azimuth 120 \
  --camera-elevation 30 \
  --off-screen \
  --outdir outputs \
  --screenshot outputs/torus_primal_dual.pdf
"""
