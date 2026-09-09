#!/usr/bin/env python3
"""Projection sur la sphère des maillages primal et dual barycentrique.

Les maillages classiques sont d'abord construits avec la même connectivité
que dans ``sphere_dual_barycentric_gmsh.py``. Ils sont ensuite élevés à un
ordre de Lagrange donné et tous leurs nœuds géométriques sont projetés
radialement sur la sphère lisse. Les éléments d'ordre supérieur décrivent
donc des arêtes et des cellules courbes (à la précision de l'ordre choisi).

Sorties :
  sphere_primal_curved.msh
  sphere_dual_barycentric_curved.msh
  sphere_diamonds_curved.msh
  sphere_dual_curved_cells.csv
  sphere_diamonds_curved.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import gmsh

from sphere_dual_barycentric_gmsh import (
    Edge,
    Point,
    Triangle,
    barycenter,
    canonical_edge,
    create_primal_mesh,
    midpoint,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Projections courbes des maillages primal et dual sur une sphère."
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--lc", type=float, default=0.18)
    parser.add_argument(
        "--projection-order",
        type=int,
        default=3,
        help="Ordre des éléments courbes de Lagrange (défaut : 3).",
    )
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument(
        "--keep-flat-primal",
        action="store_true",
        help="Conserve aussi le maillage intermédiaire sphere_primal_flat.msh.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Ne lance pas la visualisation PyVista.",
    )
    parser.add_argument(
        "--off-screen",
        action="store_true",
        help="Produit l'image sans ouvrir de fenêtre PyVista.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help=(
            "Chemin du rendu PNG/JPEG/TIFF/BMP ou PDF/SVG/EPS/PS "
            "(défaut : OUTDIR/sphere_primal_dual_curved.png)."
        ),
    )
    parser.add_argument(
        "--curve-samples",
        type=int,
        default=24,
        help="Nombre de segments d'affichage par arête courbe (défaut : 24).",
    )
    parser.add_argument("--primal-width", type=float, default=2.0)
    parser.add_argument("--dual-width", type=float, default=3.0)
    parser.add_argument("--diamond-width", type=float, default=2.0)
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
    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument(
        "--gui",
        dest="gui",
        action="store_true",
        help="Ouvre aussi le fichier dual dans la fenêtre Gmsh.",
    )
    gui_group.add_argument(
        "--no-gui",
        dest="gui",
        action="store_false",
        help="Génère seulement les fichiers, sans ouvrir de fenêtre.",
    )
    parser.set_defaults(gui=False)
    return parser.parse_args()


def radial_projection(point: Point, radius: float) -> Point:
    length = math.sqrt(sum(coordinate * coordinate for coordinate in point))
    if length <= 1.0e-15 * radius:
        raise RuntimeError("Impossible de projeter le centre de la sphère.")
    factor = radius / length
    return tuple(factor * coordinate for coordinate in point)  # type: ignore[return-value]


def project_all_mesh_nodes(radius: float) -> None:
    """Projette chaque nœud, y compris les nœuds géométriques d'ordre élevé."""
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    for index, node_tag in enumerate(node_tags):
        point = (
            float(coordinates[3 * index]),
            float(coordinates[3 * index + 1]),
            float(coordinates[3 * index + 2]),
        )
        projected = radial_projection(point, radius)
        gmsh.model.mesh.setNode(int(node_tag), list(projected), [])


def add_nodes(surface_tag: int, points: Dict[int, Point]) -> None:
    node_tags = sorted(points)
    coordinates = [value for tag in node_tags for value in points[tag]]
    gmsh.model.mesh.addNodes(2, surface_tag, node_tags, coordinates)


def create_curved_primal(
    primal_points: Dict[int, Point],
    triangles: Sequence[Triangle],
    radius: float,
    order: int,
    output_file: Path,
) -> None:
    """Crée proj(Sigma_h) avec la connectivité du primal triangulaire."""
    gmsh.clear()
    gmsh.model.add("sphere_primal_curved")
    surface_tag = gmsh.model.addDiscreteEntity(2)

    old_tags = sorted(primal_points)
    old_to_new = {tag: index + 1 for index, tag in enumerate(old_tags)}
    points = {old_to_new[tag]: primal_points[tag] for tag in old_tags}
    add_nodes(surface_tag, points)

    element_tags = list(range(1, len(triangles) + 1))
    connectivity = [old_to_new[tag] for triangle in triangles for tag in triangle]
    gmsh.model.mesh.addElementsByType(surface_tag, 2, element_tags, connectivity)

    physical_tag = gmsh.model.addPhysicalGroup(2, [surface_tag])
    gmsh.model.setPhysicalName(2, physical_tag, "primal_curved")
    gmsh.model.mesh.setOrder(order)
    project_all_mesh_nodes(radius)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(str(output_file))


def create_curved_dual(
    primal_points: Dict[int, Point],
    triangles: Sequence[Triangle],
    radius: float,
    order: int,
    output_file: Path,
    cell_map_file: Path,
) -> None:
    """Crée la projection du dual barycentrique classique sur la sphère."""
    gmsh.clear()
    gmsh.model.add("sphere_dual_barycentric_curved")
    surface_tag = gmsh.model.addDiscreteEntity(2)

    dual_points: Dict[int, Point] = {}
    primal_nodes: Dict[int, int] = {}
    midpoint_nodes: Dict[Edge, int] = {}
    barycenter_nodes: Dict[int, int] = {}
    next_node = 1

    for primal_tag in sorted(primal_points):
        primal_nodes[primal_tag] = next_node
        dual_points[next_node] = primal_points[primal_tag]
        next_node += 1

    edges = sorted(
        {
            canonical_edge(a, b)
            for a, b, c in triangles
            for a, b in ((a, b), (b, c), (c, a))
        }
    )
    for edge in edges:
        midpoint_nodes[edge] = next_node
        dual_points[next_node] = midpoint(
            primal_points[edge[0]], primal_points[edge[1]]
        )
        next_node += 1

    for face_index, (a, b, c) in enumerate(triangles):
        barycenter_nodes[face_index] = next_node
        dual_points[next_node] = barycenter(
            primal_points[a], primal_points[b], primal_points[c]
        )
        next_node += 1

    add_nodes(surface_tag, dual_points)
    element_tags: List[int] = []
    connectivity: List[int] = []
    rows: List[Tuple[int, int, int]] = []
    next_element = 1

    for face_index, (a, b, c) in enumerate(triangles):
        g = barycenter_nodes[face_index]
        mab = midpoint_nodes[canonical_edge(a, b)]
        mbc = midpoint_nodes[canonical_edge(b, c)]
        mca = midpoint_nodes[canonical_edge(c, a)]
        local_quads = (
            (a, (primal_nodes[a], mab, g, mca)),
            (b, (primal_nodes[b], mbc, g, mab)),
            (c, (primal_nodes[c], mca, g, mbc)),
        )
        for primal_vertex, quad in local_quads:
            element_tags.append(next_element)
            connectivity.extend(quad)
            rows.append((primal_vertex, next_element, face_index + 1))
            next_element += 1

    gmsh.model.mesh.addElementsByType(surface_tag, 3, element_tags, connectivity)
    physical_tag = gmsh.model.addPhysicalGroup(2, [surface_tag])
    gmsh.model.setPhysicalName(2, physical_tag, "dual_barycentric_curved")

    # Les nouveaux nœuds sont d'abord interpolés dans chaque quadrilatère plat,
    # puis relevés sur la surface lisse par la projection radiale.
    gmsh.model.mesh.setOrder(order)
    project_all_mesh_nodes(radius)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(str(output_file))

    with cell_map_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["primal_vertex_tag", "dual_element_tag", "primal_face_index"]
        )
        writer.writerows(rows)


def create_curved_diamonds(
    primal_points: Dict[int, Point],
    triangles: Sequence[Triangle],
    radius: float,
    order: int,
    output_file: Path,
    cell_map_file: Path,
) -> None:
    """Crée un diamant courbe A--G_K--B--G_L par arête primale A--B."""
    gmsh.clear()
    gmsh.model.add("sphere_diamonds_curved")
    surface_tag = gmsh.model.addDiscreteEntity(2)

    flat_barycenters = [
        barycenter(*(primal_points[tag] for tag in triangle))
        for triangle in triangles
    ]
    adjacent_faces: Dict[Edge, List[int]] = {}
    for face_index, (a, b, c) in enumerate(triangles):
        for edge in ((a, b), (b, c), (c, a)):
            adjacent_faces.setdefault(canonical_edge(*edge), []).append(face_index)

    # Les nœuds sont volontairement partagés entre les diamants : un sommet
    # primal ou un barycentre projeté garde ainsi un unique tag Gmsh.
    mesh_points: Dict[int, Point] = {}
    primal_node: Dict[int, int] = {}
    face_node: Dict[int, int] = {}
    next_node = 1
    for tag in sorted(primal_points):
        primal_node[tag] = next_node
        mesh_points[next_node] = primal_points[tag]
        next_node += 1
    for face_index, center in enumerate(flat_barycenters):
        face_node[face_index] = next_node
        mesh_points[next_node] = center
        next_node += 1
    add_nodes(surface_tag, mesh_points)

    element_tags: List[int] = []
    connectivity: List[int] = []
    rows: List[Tuple[int, int, int, int, int]] = []
    for element_tag, edge in enumerate(sorted(adjacent_faces), start=1):
        faces = adjacent_faces[edge]
        if len(faces) != 2:
            raise RuntimeError(
                f"L'arête {edge} possède {len(faces)} faces adjacentes au lieu de 2."
            )
        a, b = edge
        k, l = faces
        # Ordre de bord ; les diagonales sont A--B et G_K--G_L.
        connectivity.extend(
            (primal_node[a], face_node[k], primal_node[b], face_node[l])
        )
        element_tags.append(element_tag)
        rows.append((element_tag, a, b, k + 1, l + 1))

    gmsh.model.mesh.addElementsByType(surface_tag, 3, element_tags, connectivity)
    physical_tag = gmsh.model.addPhysicalGroup(2, [surface_tag])
    gmsh.model.setPhysicalName(2, physical_tag, "diamonds_curved")
    gmsh.model.mesh.setOrder(order)
    project_all_mesh_nodes(radius)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(str(output_file))

    with cell_map_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["diamond_tag", "primal_a", "primal_b", "face_k", "face_l"]
        )
        writer.writerows(rows)


def render_curved_primal_and_dual(
    primal_points: Dict[int, Point],
    triangles: Sequence[Triangle],
    radius: float,
    screenshot_file: Path,
    off_screen: bool,
    curve_samples: int,
    primal_width: float,
    dual_width: float,
    diamond_width: float,
    camera_azimuth: float,
    camera_elevation: float,
) -> None:
    """Affiche la sphère lisse et les projections courbes des deux réseaux."""
    try:
        import numpy as np
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "Le rendu nécessite numpy et pyvista. Installez-les avec : "
            "python3 -m pip install numpy pyvista"
        ) from exc

    # Un très léger décalage radial garde les courbes visibles devant la sphère.
    display_radius = radius * (1.0 + 2.0e-3)

    def sampled_projected_segment(p: Point, q: Point) -> List[Point]:
        result = []
        for index in range(curve_samples + 1):
            t = index / curve_samples
            flat_point = tuple(
                (1.0 - t) * p[axis] + t * q[axis] for axis in range(3)
            )
            result.append(radial_projection(flat_point, display_radius))
        return result

    def wire_polydata(segments: Sequence[Tuple[Point, Point]]):
        line_points: List[Point] = []
        lines: List[int] = []
        for p, q in segments:
            samples = sampled_projected_segment(p, q)
            first = len(line_points)
            line_points.extend(samples)
            lines.extend((len(samples), *(first + i for i in range(len(samples)))))
        return pv.PolyData(
            np.asarray(line_points, dtype=float),
            lines=np.asarray(lines, dtype=np.int64),
        )

    primal_edges = sorted(
        {
            canonical_edge(a, b)
            for a, b, c in triangles
            for a, b in ((a, b), (b, c), (c, a))
        }
    )
    primal_segments = [
        (primal_points[edge[0]], primal_points[edge[1]]) for edge in primal_edges
    ]

    # Arêtes du dual classique : barycentre de face -- milieu d'arête.
    # La projection est appliquée à tout le segment, pas seulement aux extrémités.
    dual_segments: List[Tuple[Point, Point]] = []
    diamond_segments: List[Tuple[Point, Point]] = []
    for a, b, c in triangles:
        pa, pb, pc = primal_points[a], primal_points[b], primal_points[c]
        g = barycenter(pa, pb, pc)
        dual_segments.extend(
            (
                (g, midpoint(pa, pb)),
                (g, midpoint(pb, pc)),
                (g, midpoint(pc, pa)),
            )
        )
        # Dans chaque face, ce sont trois côtés appartenant à trois diamants.
        diamond_segments.extend(((pa, g), (pb, g), (pc, g)))

    primal_wire = wire_polydata(primal_segments)
    dual_wire = wire_polydata(dual_segments)
    diamond_wire = wire_polydata(diamond_segments)
    sphere = pv.Sphere(
        radius=radius,
        theta_resolution=max(120, 4 * curve_samples),
        phi_resolution=max(120, 4 * curve_samples),
    )

    screenshot_file.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=off_screen, window_size=(1600, 1600))
    plotter.set_background("white")
    plotter.add_mesh(
        sphere,
        color="light_viridian", # cornflower_blue
        opacity=1.0,
        smooth_shading=True,
        show_edges=False,
        lighting=True,
    )
    """
    plotter.add_mesh(
        primal_wire,
        color="blue",
        line_width=primal_width,
        render_lines_as_tubes=False,
        lighting=True,
    )
    plotter.add_mesh(
        dual_wire,
        color="red",
        line_width=dual_width,
        render_lines_as_tubes=False,
        lighting=True,
    )
    """
    plotter.add_mesh(
        diamond_wire,
        color="green",
        line_width=diamond_width,
        render_lines_as_tubes=False,
        lighting=True,
    )
    # Position sphérique explicite de la caméra.
    azimuth = math.radians(camera_azimuth)
    elevation = math.radians(camera_elevation)
    camera_distance = 3.0 * display_radius
    plotter.camera.focal_point = (0.0, 0.0, 0.0)
    plotter.camera.position = (
        camera_distance * math.cos(elevation) * math.cos(azimuth),
        camera_distance * math.cos(elevation) * math.sin(azimuth),
        camera_distance * math.sin(elevation),
    )
    plotter.camera.up = (0.0, 0.0, 1.0)

    # Recadrage serré conservant les angles. zoom("tight") n'est pas utilisé,
    # car PyVista réoriente alors la caméra suivant le plan xy.
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
            np.asarray(sphere.points),
            np.asarray(primal_wire.points),
            np.asarray(dual_wire.points),
            np.asarray(diamond_wire.points),
        )
    )
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
    extension = screenshot_file.suffix.lower()

    if extension in graphic_extensions:
        plotter.show(auto_close=False)
        plotter.save_graphic(
            screenshot_file,
            title="Maillages courbes primal et dual de la sphère",
            raster=True,
        )
        plotter.close()
    elif extension in raster_extensions:
        plotter.show(screenshot=str(screenshot_file), auto_close=True)
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
    if args.projection_order < 2 or args.projection_order > 10:
        raise ValueError("--projection-order doit être compris entre 2 et 10.")
    if args.curve_samples < 2:
        raise ValueError("--curve-samples doit être supérieur ou égal à 2.")
    if not math.isfinite(args.camera_azimuth):
        raise ValueError("--camera-azimuth doit être un nombre fini.")
    if not math.isfinite(args.camera_elevation) or not (
        -89.0 < args.camera_elevation < 89.0
    ):
        raise ValueError(
            "--camera-elevation doit être comprise entre -89 et 89 degrés."
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    flat_primal_file = args.outdir / "sphere_primal_flat.msh"
    primal_file = args.outdir / "sphere_primal_curved.msh"
    dual_file = args.outdir / "sphere_dual_barycentric_curved.msh"
    diamond_file = args.outdir / "sphere_diamonds_curved.msh"
    map_file = args.outdir / "sphere_dual_curved_cells.csv"
    diamond_map_file = args.outdir / "sphere_diamonds_curved.csv"
    screenshot_file = args.screenshot or (
        args.outdir / "sphere_primal_dual_curved.png"
    )

    gmsh.initialize([])
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        points, triangles = create_primal_mesh(
            args.radius, args.lc, flat_primal_file
        )
        create_curved_primal(
            points, triangles, args.radius, args.projection_order, primal_file
        )
        create_curved_dual(
            points,
            triangles,
            args.radius,
            args.projection_order,
            dual_file,
            map_file,
        )
        create_curved_diamonds(
            points,
            triangles,
            args.radius,
            args.projection_order,
            diamond_file,
            diamond_map_file,
        )

        if not args.keep_flat_primal:
            flat_primal_file.unlink(missing_ok=True)

        print(f"Maillage primal courbe : {primal_file}")
        print(f"Maillage dual courbe   : {dual_file}")
        print(f"Diamants courbes       : {diamond_file}")
        print(f"Correspondance duale   : {map_file}")
        print(f"Correspondance diamants: {diamond_map_file}")
        print(f"Ordre géométrique      : {args.projection_order}")
        print(f"Triangles primaux      : {len(triangles)}")
        print(f"Quadrilatères duaux    : {3 * len(triangles)}")
        print(
            "Diamants courbes       : "
            f"{3 * len(triangles) // 2}"
        )

        if not args.no_render:
            render_curved_primal_and_dual(
                points,
                triangles,
                args.radius,
                screenshot_file,
                args.off_screen,
                args.curve_samples,
                args.primal_width,
                args.dual_width,
                args.diamond_width,
                args.camera_azimuth,
                args.camera_elevation,
            )
            print(f"Rendu PyVista          : {screenshot_file}")

        if args.gui:
            gmsh.open(str(dual_file))
            gmsh.fltk.run()
    finally:
        gmsh.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python3 sphere_curved_primal_dual_gmsh.py --radius 1 --lc 0.5 --projection-order 3 --outdir outputs
# python3 sphere_curved_primal_dual_gmsh.py --diamond-width 3 --outdir outputs

"""
python3 sphere_curved_primal_dual_gmsh.py \
  --radius 1 \
  --lc 1.2 \
  --primal-width 12 \
  --dual-width 12 \
  --diamond-width 12 \
  --projection-order 3 \
  --curve-samples 24 \
  --camera-azimuth 115 \
  --camera-elevation 15 \
  --off-screen \
  --outdir outputs \
  --screenshot outputs/sphere_primal_dual_curved.pdf
"""