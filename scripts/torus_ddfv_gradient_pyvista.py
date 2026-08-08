#!/usr/bin/env python3
"""Champ scalaire oscillant et approximation DDFV du gradient sur un tore.

Pour les angles toroidaux theta et phi, le champ test est

  u(theta, phi) = sin(3 theta) cos(2 phi) + 0.35 sin(theta + 3 phi).

La fenêtre PyVista montre le champ sur le tore lisse et, à côté, un vecteur
gradient DDFV constant sur chaque diamant plat.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import gmsh
import numpy as np

from torus_dual_barycentric_gmsh import (
    Edge,
    Point,
    Triangle,
    canonical_edge,
    create_primal_mesh,
)


Diamond = Tuple[Point, Point, Point, Point]
DiamondData = Tuple[Edge, Tuple[int, int], Diamond]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Champ oscillant et gradient discret DDFV sur un tore."
    )
    parser.add_argument("--major-radius", type=float, default=1.5)
    parser.add_argument("--minor-radius", type=float, default=0.5)
    parser.add_argument("--lc", type=float, default=0.25)
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help=(
            "Chemin de base facultatif : les suffixes _field et _gradient "
            "sont ajoutés pour créer deux fichiers distincts."
        ),
    )
    parser.add_argument(
        "--field-screenshot",
        type=Path,
        default=None,
        help="Chemin du rendu séparé du champ scalaire.",
    )
    parser.add_argument(
        "--gradient-screenshot",
        type=Path,
        default=None,
        help="Chemin du rendu séparé du gradient DDFV.",
    )
    parser.add_argument(
        "--off-screen",
        action="store_true",
        help="Crée uniquement le rendu, sans fenêtre interactive PyVista.",
    )
    parser.add_argument("--arrow-scale", type=float, default=0.24)
    parser.add_argument("--torus-resolution", type=int, default=180)
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


def torus_angles(point: Point | np.ndarray, major_radius: float) -> Tuple[float, float]:
    x, y, z = np.asarray(point, dtype=float)
    theta = math.atan2(float(y), float(x))
    rho = math.hypot(float(x), float(y))
    phi = math.atan2(float(z), rho - major_radius)
    return theta, phi


def torus_projection(
    point: Point | np.ndarray, major_radius: float, minor_radius: float
) -> Point:
    """Projection normale sur le tore annulaire centré autour de Oz."""
    p = np.asarray(point, dtype=float)
    rho = math.hypot(float(p[0]), float(p[1]))
    if rho <= 1.0e-14 * major_radius:
        raise RuntimeError("Projection indéfinie sur l'axe du tore.")
    centerline = np.asarray(
        (major_radius * p[0] / rho, major_radius * p[1] / rho, 0.0)
    )
    radial = p - centerline
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm <= 1.0e-14 * minor_radius:
        raise RuntimeError("Projection indéfinie sur la ligne centrale du tore.")
    projected = centerline + minor_radius * radial / radial_norm
    return tuple(float(value) for value in projected)


def scalar_field(point: Point | np.ndarray, major_radius: float) -> float:
    theta, phi = torus_angles(point, major_radius)
    return float(
        0.7 * math.sin(3.0 * theta) * math.cos(2.0 * phi)
        + 0.3 * math.sin(theta + 3.0 * phi)
    )


def exact_surface_gradient(
    point: Point | np.ndarray, major_radius: float, minor_radius: float
) -> np.ndarray:
    theta, phi = torus_angles(point, major_radius)
    u_theta = (
        3.0 * math.cos(3.0 * theta) * math.cos(2.0 * phi)
        + 0.35 * math.cos(theta + 3.0 * phi)
    )
    u_phi = (
        -2.0 * math.sin(3.0 * theta) * math.sin(2.0 * phi)
        + 1.05 * math.cos(theta + 3.0 * phi)
    )
    e_theta = np.asarray((-math.sin(theta), math.cos(theta), 0.0))
    e_phi = np.asarray(
        (
            -math.sin(phi) * math.cos(theta),
            -math.sin(phi) * math.sin(theta),
            math.cos(phi),
        )
    )
    return (
        u_theta / (major_radius + minor_radius * math.cos(phi)) * e_theta
        + u_phi / minor_radius * e_phi
    )


def build_flat_diamonds(
    points: Dict[int, Point],
    triangles: Sequence[Triangle],
    major_radius: float,
    minor_radius: float,
) -> Tuple[List[DiamondData], List[Point]]:
    """Même construction plane que pour les diamants de la sphère."""
    face_centers: List[Point] = []
    for a, b, c in triangles:
        center = tuple(
            (points[a][axis] + points[b][axis] + points[c][axis]) / 3.0
            for axis in range(3)
        )
        face_centers.append(torus_projection(center, major_radius, minor_radius))

    adjacent_faces: Dict[Edge, List[int]] = defaultdict(list)
    for face_index, (a, b, c) in enumerate(triangles):
        for edge in ((a, b), (b, c), (c, a)):
            adjacent_faces[canonical_edge(*edge)].append(face_index)

    diamonds: List[DiamondData] = []
    tolerance = 1.0e-13 * minor_radius * minor_radius
    for edge in sorted(adjacent_faces):
        faces = adjacent_faces[edge]
        if len(faces) != 2:
            raise RuntimeError(
                f"L'arête {edge} possède {len(faces)} faces adjacentes au lieu de 2."
            )
        a = np.asarray(points[edge[0]], dtype=float)
        b = np.asarray(points[edge[1]], dtype=float)
        g = np.asarray(face_centers[faces[0]], dtype=float)
        h = np.asarray(face_centers[faces[1]], dtype=float)
        normal = np.cross(b - a, h - g)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= tolerance:
            raise RuntimeError(f"Plan du diamant dégénéré pour l'arête {edge}.")
        normal /= normal_norm
        plane_origin = 0.25 * (a + b + g + h)

        def project_on_plane(p: np.ndarray) -> Point:
            q = p - np.dot(p - plane_origin, normal) * normal
            return tuple(float(value) for value in q)

        vertices = tuple(project_on_plane(p) for p in (a, g, b, h))
        diamonds.append((edge, (faces[0], faces[1]), vertices))
    return diamonds, face_centers


def discrete_gradient_on_diamond(
    diamond: DiamondData,
    primal_points: Dict[int, Point],
    face_centers: Sequence[Point],
    major_radius: float,
) -> np.ndarray:
    edge, faces, vertices = diamond
    a_flat, g_flat, b_flat, h_flat = map(
        lambda p: np.asarray(p, dtype=float), vertices
    )
    sigma = b_flat - a_flat
    dual_sigma = h_flat - g_flat
    sigma_length = float(np.linalg.norm(sigma))
    dual_length = float(np.linalg.norm(dual_sigma))
    directions = np.vstack((dual_sigma / dual_length, sigma / sigma_length))
    differences = np.asarray(
        (
            (
                scalar_field(face_centers[faces[1]], major_radius)
                - scalar_field(face_centers[faces[0]], major_radius)
            )
            / dual_length,
            (
                scalar_field(primal_points[edge[1]], major_radius)
                - scalar_field(primal_points[edge[0]], major_radius)
            )
            / sigma_length,
        )
    )
    gram = directions @ directions.T
    if abs(float(np.linalg.det(gram))) <= 1.0e-14:
        raise RuntimeError(f"Directions presque colinéaires pour l'arête {edge}.")
    return directions.T @ np.linalg.solve(gram, differences)


def compute_gradients(
    diamonds: Sequence[DiamondData],
    primal_points: Dict[int, Point],
    face_centers: Sequence[Point],
    major_radius: float,
    minor_radius: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers, gradients, errors = [], [], []
    for diamond in diamonds:
        center = np.mean(np.asarray(diamond[2], dtype=float), axis=0)
        gradient = discrete_gradient_on_diamond(
            diamond, primal_points, face_centers, major_radius
        )
        smooth_center = np.asarray(
            torus_projection(center, major_radius, minor_radius)
        )
        exact = exact_surface_gradient(smooth_center, major_radius, minor_radius)
        centers.append(center)
        gradients.append(gradient)
        errors.append(float(np.linalg.norm(gradient - exact)))
    return np.asarray(centers), np.asarray(gradients), np.asarray(errors)


def write_outputs(
    mesh_file: Path,
    map_file: Path,
    gradient_file: Path,
    diamonds: Sequence[DiamondData],
    centers: np.ndarray,
    gradients: np.ndarray,
    errors: np.ndarray,
) -> None:
    gmsh.clear()
    gmsh.model.add("torus_flat_diamonds")
    surface_tag = gmsh.model.addDiscreteEntity(2)
    node_tags, coordinates, element_tags, connectivity = [], [], [], []
    map_rows = []
    for element_tag, (edge, faces, vertices) in enumerate(diamonds, start=1):
        local_nodes = []
        for vertex in vertices:
            tag = len(node_tags) + 1
            node_tags.append(tag)
            coordinates.extend(vertex)
            local_nodes.append(tag)
        element_tags.append(element_tag)
        connectivity.extend(local_nodes)
        map_rows.append((element_tag, *edge, faces[0] + 1, faces[1] + 1))
    gmsh.model.mesh.addNodes(2, surface_tag, node_tags, coordinates)
    gmsh.model.mesh.addElementsByType(
        surface_tag, 3, element_tags, connectivity
    )
    physical = gmsh.model.addPhysicalGroup(2, [surface_tag])
    gmsh.model.setPhysicalName(2, physical, "flat_diamonds")
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(str(mesh_file))

    with map_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("diamond_tag", "primal_a", "primal_b", "face_k", "face_l"))
        writer.writerows(map_rows)
    with gradient_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "diamond_tag", "center_x", "center_y", "center_z",
                "grad_x", "grad_y", "grad_z", "error_to_exact_gradient",
            )
        )
        for tag, (center, gradient, error) in enumerate(
            zip(centers, gradients, errors), start=1
        ):
            writer.writerow((tag, *center, *gradient, error))


def smooth_torus_polydata(
    major_radius: float, minor_radius: float, resolution: int
):
    import pyvista as pv

    points: List[Tuple[float, float, float]] = []
    values: List[float] = []
    for i in range(resolution):
        theta = 2.0 * math.pi * i / resolution
        for j in range(resolution):
            phi = 2.0 * math.pi * j / resolution
            point = (
                (major_radius + minor_radius * math.cos(phi)) * math.cos(theta),
                (major_radius + minor_radius * math.cos(phi)) * math.sin(theta),
                minor_radius * math.sin(phi),
            )
            points.append(point)
            values.append(scalar_field(point, major_radius))
    faces: List[int] = []
    for i in range(resolution):
        for j in range(resolution):
            a = i * resolution + j
            b = ((i + 1) % resolution) * resolution + j
            c = ((i + 1) % resolution) * resolution + (j + 1) % resolution
            d = i * resolution + (j + 1) % resolution
            faces.extend((4, a, b, c, d))
    mesh = pv.PolyData(
        np.asarray(points), faces=np.asarray(faces, dtype=np.int64)
    )
    mesh["u"] = np.asarray(values)
    return mesh


def render(
    major_radius: float,
    minor_radius: float,
    diamonds: Sequence[DiamondData],
    centers: np.ndarray,
    gradients: np.ndarray,
    errors: np.ndarray,
    field_screenshot: Path,
    gradient_screenshot: Path,
    off_screen: bool,
    arrow_scale: float,
    resolution: int,
    camera_azimuth: float,
    camera_elevation: float,
) -> None:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("Le rendu nécessite PyVista.") from exc

    torus = smooth_torus_polydata(major_radius, minor_radius, resolution)
    diamond_points, diamond_faces = [], []
    for _, _, vertices in diamonds:
        first = len(diamond_points)
        diamond_points.extend(vertices)
        diamond_faces.extend((4, first, first + 1, first + 2, first + 3))
    diamond_mesh = pv.PolyData(
        np.asarray(diamond_points), faces=np.asarray(diamond_faces, dtype=np.int64)
    )
    diamond_mesh.cell_data["erreur gradient"] = errors

    vector_points = pv.PolyData(centers)
    vector_points["gradient DDFV"] = gradients
    maximum_norm = max(float(np.linalg.norm(gradients, axis=1).max()), 1.0e-15)
    arrows = vector_points.glyph(
        orient="gradient DDFV",
        scale="gradient DDFV",
        factor=arrow_scale * major_radius / maximum_norm,
        geom=pv.Arrow(tip_resolution=16, shaft_resolution=12),
    )

    azimuth = math.radians(camera_azimuth)
    elevation = math.radians(camera_elevation)
    scene_scale = max(float(np.linalg.norm(point)) for point in torus.points)
    camera_distance = 3.0 * scene_scale
    camera_position = np.asarray(
        (
            camera_distance * math.cos(elevation) * math.cos(azimuth),
            camera_distance * math.cos(elevation) * math.sin(azimuth),
            camera_distance * math.sin(elevation),
        )
    )
    focal_point = np.zeros(3)
    view_direction = focal_point - camera_position
    view_direction /= np.linalg.norm(view_direction)
    view_up = np.asarray((0.0, 0.0, 1.0))
    camera_right = np.cross(view_direction, view_up)
    camera_right /= np.linalg.norm(camera_right)
    view_up = np.cross(camera_right, view_direction)
    view_up /= np.linalg.norm(view_up)
    camera = [tuple(camera_position), tuple(focal_point), tuple(view_up)]

    field_screenshot.parent.mkdir(parents=True, exist_ok=True)
    gradient_screenshot.parent.mkdir(parents=True, exist_ok=True)

    field_plotter = pv.Plotter(off_screen=off_screen, window_size=(1600, 1600))
    field_plotter.set_background("white")
    # field_plotter.add_text(
    #     "Champ scalaire oscillant sur le tore lisse", font_size=13
    # )
    field_plotter.add_mesh(
        torus, scalars="u", cmap="coolwarm", smooth_shading=True,
        show_scalar_bar=False,
    )
    field_plotter.camera_position = camera

    gradient_plotter = pv.Plotter(off_screen=off_screen, window_size=(1600, 1600))
    gradient_plotter.set_background("white")
    # gradient_plotter.add_text(
    #     "Gradient DDFV constant par diamant plat", font_size=13
    # )
    gradient_plotter.add_mesh(
        diamond_mesh, scalars="erreur gradient", cmap="viridis", opacity=1,
        show_edges=True, edge_color="gray", line_width=1.0,
        show_scalar_bar=False,
    )
    gradient_plotter.add_mesh(arrows, color="black", lighting=True)
    gradient_plotter.add_silhouette(
        arrows,
        color="white",
        line_width=5.0,
    )
    gradient_plotter.camera_position = camera

    raster_extensions = {".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff"}
    graphic_extensions = {".pdf", ".svg", ".eps", ".ps", ".tex"}

    def configure_and_export(
        plotter,
        rendered_points: np.ndarray,
        output_file: Path,
        title: str,
    ) -> None:
        """Recadre et exporte une vue sans modifier son angle de caméra."""
        relative_points = rendered_points - focal_point
        half_width = float(np.max(np.abs(relative_points @ camera_right)))
        half_height = float(np.max(np.abs(relative_points @ view_up)))

        projected_aspect_ratio = half_width / half_height
        maximum_dimension = 1600
        if projected_aspect_ratio >= 1.0:
            window_width = maximum_dimension
            window_height = max(
                1, round(maximum_dimension / projected_aspect_ratio)
            )
        else:
            window_width = max(
                1, round(maximum_dimension * projected_aspect_ratio)
            )
            window_height = maximum_dimension
        plotter.window_size = (window_width, window_height)

        actual_aspect = window_width / window_height
        parallel_scale = 1.03 * max(
            half_height, half_width / actual_aspect
        )
        plotter.camera_position = camera
        plotter.camera.enable_parallel_projection()
        plotter.camera.parallel_scale = parallel_scale
        plotter.reset_camera_clipping_range()
        plotter.enable_anti_aliasing("ssaa")

        extension = output_file.suffix.lower()
        if extension in graphic_extensions:
            plotter.show(auto_close=False)
            plotter.save_graphic(output_file, title=title, raster=True)
            plotter.close()
        elif extension in raster_extensions:
            plotter.show(screenshot=str(output_file), auto_close=True)
        else:
            plotter.close()
            supported = ", ".join(
                sorted(raster_extensions | graphic_extensions)
            )
            raise ValueError(
                f"Extension de rendu non prise en charge : "
                f"{extension or '(aucune)'}. Extensions acceptées : {supported}"
            )

    configure_and_export(
        field_plotter,
        np.asarray(torus.points),
        field_screenshot,
        "Champ scalaire oscillant sur un tore",
    )
    configure_and_export(
        gradient_plotter,
        np.vstack((np.asarray(diamond_points), np.asarray(arrows.points))),
        gradient_screenshot,
        "Gradient DDFV sur les diamants d'un tore",
    )


def main() -> int:
    args = parse_arguments()
    if not (math.isfinite(args.major_radius) and args.major_radius > 0.0):
        raise ValueError("--major-radius doit être strictement positif.")
    if not (math.isfinite(args.minor_radius) and args.minor_radius > 0.0):
        raise ValueError("--minor-radius doit être strictement positif.")
    if args.major_radius <= args.minor_radius:
        raise ValueError("--major-radius doit être supérieur à --minor-radius.")
    if not (math.isfinite(args.lc) and args.lc > 0.0):
        raise ValueError("--lc doit être strictement positif.")
    if args.torus_resolution < 20:
        raise ValueError("--torus-resolution doit être au moins égal à 20.")
    if not math.isfinite(args.camera_azimuth):
        raise ValueError("--camera-azimuth doit être un nombre fini.")
    if not math.isfinite(args.camera_elevation) or not (
        -89.0 < args.camera_elevation < 89.0
    ):
        raise ValueError(
            "--camera-elevation doit être comprise entre -89 et 89 degrés."
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    primal_file = args.outdir / "torus_primal_for_gradient.msh"
    diamond_file = args.outdir / "torus_diamonds_for_gradient.msh"
    map_file = args.outdir / "torus_diamonds_for_gradient.csv"
    gradient_file = args.outdir / "torus_ddfv_gradient.csv"

    def add_output_suffix(path: Path, suffix: str) -> Path:
        extension = path.suffix or ".pdf"
        stem = path.stem if path.suffix else path.name
        return path.with_name(f"{stem}_{suffix}{extension}")

    field_screenshot = args.field_screenshot
    gradient_screenshot = args.gradient_screenshot
    if field_screenshot is None:
        field_screenshot = (
            add_output_suffix(args.screenshot, "field")
            if args.screenshot is not None
            else args.outdir / "torus_ddfv_field.pdf"
        )
    if gradient_screenshot is None:
        gradient_screenshot = (
            add_output_suffix(args.screenshot, "gradient")
            if args.screenshot is not None
            else args.outdir / "torus_ddfv_gradient.pdf"
        )

    gmsh.initialize([])
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        points, triangles = create_primal_mesh(
            args.major_radius, args.minor_radius, args.lc, primal_file
        )
        diamonds, face_centers = build_flat_diamonds(
            points, triangles, args.major_radius, args.minor_radius
        )
        centers, gradients, errors = compute_gradients(
            diamonds, points, face_centers, args.major_radius, args.minor_radius
        )
        write_outputs(
            diamond_file, map_file, gradient_file,
            diamonds, centers, gradients, errors,
        )
    finally:
        gmsh.finalize()

    render(
        args.major_radius, args.minor_radius, diamonds, centers, gradients, errors,
        field_screenshot, gradient_screenshot, args.off_screen,
        args.arrow_scale, args.torus_resolution,
        args.camera_azimuth, args.camera_elevation,
    )
    print("Champ : sin(3 theta) cos(2 phi) + 0.35 sin(theta + 3 phi)")
    print(f"Diamants             : {len(diamonds)}")
    print(f"Erreur L2 empirique  : {np.sqrt(np.mean(errors**2)):.6e}")
    print(f"Erreur e_D minimale  : {errors.min():.6e}")
    print(f"Erreur e_D maximale  : {errors.max():.6e}")
    print(f"Valeurs des gradients: {gradient_file}")
    print(f"Maillage des diamants: {diamond_file}")
    print(f"Champ scalaire       : {field_screenshot}")
    print(f"Gradient DDFV        : {gradient_screenshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python3 torus_ddfv_gradient_pyvista.py   --major-radius 1.5   --minor-radius 0.5   --lc 0.7   --outdir outputs


"""
python3 torus_ddfv_gradient_pyvista.py \
  --major-radius 1.5 \
  --minor-radius 0.5 \
  --lc 0.3 \
  --camera-azimuth 120 \
  --camera-elevation 30 \
  --off-screen \
  --outdir outputs
"""
