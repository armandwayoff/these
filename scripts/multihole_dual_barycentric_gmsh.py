#!/usr/bin/env python3
"""
Génère avec Gmsh :
  1. une surface fermée complexe comportant au moins quatre anses ;
  2. un maillage simplicial de cette surface ;
  3. son maillage dual barycentrique ;
  4. un rendu PyVista superposant primal et dual.

La géométrie est obtenue par union booléenne de quatre tores de même centre,
dont les axes sont les quatre directions diagonales d'un tétraèdre régulier.
Elle produit une surface fortement entrelacée, proche des exemples fournis.

Le dual barycentrique est construit sur la surface polyédrique triangulée :
pour chaque triangle (a,b,c), on crée les trois quadrilatères
  [a, m_ab, g, m_ca],
  [b, m_bc, g, m_ab],
  [c, m_ca, g, m_bc],
où m_ij est le milieu de l'arête (i,j) et g le barycentre du triangle.

Sorties :
  multihole_primal.msh
  multihole_dual_barycentric.msh
  multihole_dual_cells.csv
  multihole_primal_dual.png
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
            "Maillage triangulaire d'une surface multi-anses "
            "et dual barycentrique."
        )
    )
    parser.add_argument(
        "--major-radius",
        type=float,
        default=1.35,
        help="Rayon majeur de chacun des quatre tores.",
    )
    parser.add_argument(
        "--minor-radius",
        type=float,
        default=0.48,
        help="Rayon mineur de chacun des quatre tores.",
    )
    parser.add_argument(
        "--lc",
        type=float,
        default=0.15,
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
        help="Crée uniquement l'image PNG, sans fenêtre PyVista.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Chemin de l'image PNG.",
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


def normalize(vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("Un axe de tore ne peut pas être nul.")
    return tuple(value / norm for value in vector)


def create_primal_mesh(
    major_radius: float,
    minor_radius: float,
    lc: float,
    output_file: Path,
) -> Tuple[Dict[int, Point], List[Triangle]]:
    gmsh.model.add("multihole_primal")

    # Axes dirigés vers les quatre sommets d'un tétraèdre régulier.
    axes = [
        normalize((1.0, 1.0, 1.0)),
        normalize((1.0, -1.0, -1.0)),
        normalize((-1.0, 1.0, -1.0)),
        normalize((-1.0, -1.0, 1.0)),
    ]

    torus_volumes = []

    for axis in axes:
        tag = gmsh.model.occ.addTorus(
            0.0,
            0.0,
            0.0,
            major_radius,
            minor_radius,
            zAxis=list(axis),
        )
        torus_volumes.append((3, tag))

    # L'union rend la surface unique et supprime les interfaces internes.
    fused, _ = gmsh.model.occ.fuse(
        [torus_volumes[0]],
        torus_volumes[1:],
        removeObject=True,
        removeTool=True,
    )
    gmsh.model.occ.synchronize()

    volume_tags = [tag for dim, tag in fused if dim == 3]
    if not volume_tags:
        raise RuntimeError(
            "L'union booléenne des quatre tores n'a produit aucun volume."
        )

    boundary_surfaces = gmsh.model.getBoundary(
        [(3, tag) for tag in volume_tags],
        oriented=False,
        recursive=False,
    )
    surface_tags = sorted(
        {tag for dim, tag in boundary_surfaces if dim == 2}
    )

    if not surface_tags:
        raise RuntimeError("La frontière surfacique est vide.")

    physical_surface = gmsh.model.addPhysicalGroup(2, surface_tags)
    gmsh.model.setPhysicalName(
        2,
        physical_surface,
        "multihole_primal",
    )

    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    # Algorithme frontal-Delaunay pour les surfaces courbes complexes.
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

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
    gmsh.model.add("multihole_dual_barycentric")

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

    # Type Gmsh 3 : quadrilatère bilinéaire à quatre nœuds.
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


def render_primal_and_dual(
    primal_points: Dict[int, Point],
    triangles: Sequence[Triangle],
    minor_radius: float,
    screenshot_file: Path,
    off_screen: bool,
    primal_width: float,
    dual_width: float,
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

    # Normales orientées automatiquement par PyVista.
    surface = surface.compute_normals(
        point_normals=True,
        cell_normals=True,
        auto_orient_normals=True,
        consistent_normals=True,
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

    offset = 2.0e-3 * minor_radius

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
            if nm_norm != 0.0:
                nm = nm / nm_norm
            else:
                nm = ng

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
        smooth_shading=False,
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

    plotter.camera_position = "iso"
    plotter.camera.zoom(1.15)
    plotter.enable_anti_aliasing("ssaa")

    plotter.show(
        screenshot=str(screenshot_file),
        auto_close=True,
        interactive=not off_screen,
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
            "On impose --major-radius > --minor-radius."
        )

    # Pour les quatre orientations retenues, une épaisseur suffisante est
    # nécessaire afin que les tores s'intersectent et forment une seule pièce.
    if minor_radius < 0.25 * major_radius:
        raise ValueError(
            "--minor-radius est trop petit pour assurer l'intersection "
            "des quatre tores. Utilisez environ 0.30 à 0.45 fois "
            "--major-radius."
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

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    primal_file = args.outdir / "multihole_primal.msh"
    dual_file = (
        args.outdir / "multihole_dual_barycentric.msh"
    )
    cell_map_file = (
        args.outdir / "multihole_dual_cells.csv"
    )
    screenshot_file = (
        args.screenshot
        or args.outdir / "multihole_primal_dual.png"
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
                args.minor_radius,
                screenshot_file,
                args.off_screen,
                args.primal_width,
                args.dual_width,
            )
            print(f"Rendu                 : {screenshot_file}")

        if args.gui and "-nopopup" not in sys.argv:
            gmsh.fltk.run()

    finally:
        gmsh.finalize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python3 multihole_dual_barycentric_gmsh.py --major-radius 1.5 --minor-radius 0.4 --lc 0.15 --outdir outputs