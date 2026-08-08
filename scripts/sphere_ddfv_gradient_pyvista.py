#!/usr/bin/env python3
"""Visualise un champ scalaire sphérique et son gradient discret DDFV.

Le champ oscillant choisi est

    u(x, y, z) = sin(3x) cos(2y) + 0.35 sin(4z).

Pour chaque diamant plat produit par la construction de
``sphere_diamond_barycentric_gmsh.py``, le gradient constant g_D est l'unique
vecteur du plan du diamant vérifiant les deux relations de la définition
``eq:discrete_gradient`` de ``construction_of_the_discrete_operators.tex``.

La fenêtre PyVista contient :
  - à gauche, la sphère lisse colorée par u ;
  - à droite, les diamants plats et un vecteur gradient par diamant.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import gmsh
import numpy as np

from sphere_diamond_barycentric_gmsh import (
    Diamond,
    build_diamonds,
    write_diamond_mesh,
)
from sphere_dual_barycentric_gmsh import Edge, Point, Triangle, create_primal_mesh


DiamondData = Tuple[Edge, Tuple[int, int], Diamond]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Champ scalaire sur la sphère et gradient discret DDFV."
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--lc", type=float, default=0.3)
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Défaut : OUTDIR/sphere_ddfv_gradient.png.",
    )
    parser.add_argument("--off-screen", action="store_true")
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=0.28,
        help="Longueur du plus grand vecteur, en fraction du rayon.",
    )
    parser.add_argument(
        "--sphere-resolution",
        type=int,
        default=180,
        help="Résolution angulaire de la sphère PyVista.",
    )
    return parser.parse_args()


def scalar_field(point: Point | np.ndarray) -> float:
    x, y, z = np.asarray(point, dtype=float)
    return float(np.sin(3.0 * x) * np.cos(2.0 * y) + 0.35 * np.sin(4.0 * z))


def ambient_gradient(point: Point | np.ndarray) -> np.ndarray:
    """Gradient dans R^3 d'une extension régulière du champ scalaire."""
    x, y, z = np.asarray(point, dtype=float)
    return np.asarray(
        (
            3.0 * np.cos(3.0 * x) * np.cos(2.0 * y),
            -2.0 * np.sin(3.0 * x) * np.sin(2.0 * y),
            1.4 * np.cos(4.0 * z),
        ),
        dtype=float,
    )


def exact_surface_gradient(point: np.ndarray, radius: float) -> np.ndarray:
    """Gradient tangentiel exact : (I - n tensor n) grad_R3 u."""
    normal = point / np.linalg.norm(point)
    gradient = ambient_gradient(point)
    return gradient - np.dot(gradient, normal) * normal


def discrete_gradient_on_diamond(
    edge: Edge,
    faces: Tuple[int, int],
    vertices: Diamond,
    primal_points: Dict[int, Point],
    face_centers: Sequence[Point],
) -> np.ndarray:
    """Résout les deux relations directionnelles de la définition DDFV."""
    # build_diamonds renvoie les sommets dans l'ordre A, G_K, B, G_L.
    a_flat, g_flat, b_flat, h_flat = (
        np.asarray(vertex, dtype=float) for vertex in vertices
    )
    sigma = b_flat - a_flat
    dual_sigma = h_flat - g_flat
    sigma_length = float(np.linalg.norm(sigma))
    dual_length = float(np.linalg.norm(dual_sigma))
    if sigma_length <= 0.0 or dual_length <= 0.0:
        raise RuntimeError(f"Diagonale dégénérée pour l'arête {edge}.")

    # Valeurs des inconnues primales (centres de faces) et duales (sommets).
    u_k = scalar_field(face_centers[faces[0]])
    u_l = scalar_field(face_centers[faces[1]])
    u_k_star = scalar_field(primal_points[edge[0]])
    u_l_star = scalar_field(primal_points[edge[1]])

    directions = np.vstack((dual_sigma / dual_length, sigma / sigma_length))
    directional_derivatives = np.asarray(
        ((u_l - u_k) / dual_length, (u_l_star - u_k_star) / sigma_length)
    )

    # La solution de norme minimale de ce système 2x3 appartient au plan
    # engendré par sigma et dual_sigma : c'est donc l'unique gradient du plan.
    gram = directions @ directions.T
    if abs(float(np.linalg.det(gram))) <= 1.0e-14:
        raise RuntimeError(f"Directions presque colinéaires pour l'arête {edge}.")
    return directions.T @ np.linalg.solve(gram, directional_derivatives)


def compute_gradients(
    diamonds: Sequence[DiamondData],
    primal_points: Dict[int, Point],
    face_centers: Sequence[Point],
    radius: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers: List[np.ndarray] = []
    gradients: List[np.ndarray] = []
    errors: List[float] = []
    for edge, faces, vertices in diamonds:
        center = np.mean(np.asarray(vertices, dtype=float), axis=0)
        gradient = discrete_gradient_on_diamond(
            edge, faces, vertices, primal_points, face_centers
        )
        sphere_point = radius * center / np.linalg.norm(center)
        exact = exact_surface_gradient(sphere_point, radius)
        centers.append(center)
        gradients.append(gradient)
        errors.append(float(np.linalg.norm(gradient - exact)))
    return np.asarray(centers), np.asarray(gradients), np.asarray(errors)


def write_gradient_csv(
    output_file: Path,
    diamonds: Sequence[DiamondData],
    centers: np.ndarray,
    gradients: np.ndarray,
    errors: np.ndarray,
) -> None:
    with output_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "diamond_tag",
                "primal_a",
                "primal_b",
                "center_x",
                "center_y",
                "center_z",
                "grad_x",
                "grad_y",
                "grad_z",
                "error_to_exact_surface_gradient",
            ]
        )
        for tag, ((edge, _, _), center, gradient, error) in enumerate(
            zip(diamonds, centers, gradients, errors), start=1
        ):
            writer.writerow((tag, *edge, *center, *gradient, error))


def diamond_polydata(diamonds: Sequence[DiamondData]):
    import pyvista as pv

    points: List[Point] = []
    faces: List[int] = []
    for _, _, vertices in diamonds:
        first = len(points)
        points.extend(vertices)
        faces.extend((4, first, first + 1, first + 2, first + 3))
    return pv.PolyData(
        np.asarray(points, dtype=float), faces=np.asarray(faces, dtype=np.int64)
    )


def render(
    radius: float,
    diamonds: Sequence[DiamondData],
    centers: np.ndarray,
    gradients: np.ndarray,
    errors: np.ndarray,
    screenshot: Path,
    off_screen: bool,
    arrow_scale: float,
    sphere_resolution: int,
) -> None:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "Le rendu nécessite PyVista : python3 -m pip install pyvista"
        ) from exc

    sphere = pv.Sphere(
        radius=radius,
        theta_resolution=sphere_resolution,
        phi_resolution=sphere_resolution,
    )
    x, y, z = sphere.points.T
    sphere["u"] = np.sin(3.0 * x) * np.cos(2.0 * y) + 0.35 * np.sin(4.0 * z)

    flat_diamonds = diamond_polydata(diamonds)
    # Une valeur par cellule permet de repérer les zones les moins précises.
    flat_diamonds.cell_data["erreur gradient"] = errors

    vector_points = pv.PolyData(centers)
    vector_points["gradient DDFV"] = gradients
    maximum_norm = max(float(np.linalg.norm(gradients, axis=1).max()), 1.0e-15)
    arrows = vector_points.glyph(
        orient="gradient DDFV",
        scale="gradient DDFV",
        factor=arrow_scale * radius / maximum_norm,
        geom=pv.Arrow(tip_resolution=16, shaft_resolution=12),
    )

    screenshot.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(
        shape=(1, 2), off_screen=off_screen, window_size=(2000, 1000)
    )
    plotter.set_background("white")

    plotter.subplot(0, 0)
    plotter.add_text("Champ scalaire u sur la sphère lisse", font_size=13)
    plotter.add_mesh(
        sphere,
        scalars="u",
        cmap="coolwarm",
        smooth_shading=True,
        scalar_bar_args={"title": "u(x,y,z)"},
    )
    plotter.camera_position = "iso"

    plotter.subplot(0, 1)
    plotter.add_text("Gradient DDFV constant par diamant plat", font_size=13)
    plotter.add_mesh(
        flat_diamonds,
        scalars="erreur gradient",
        cmap="viridis",
        opacity=1,
        show_edges=True,
        edge_color="gray",
        line_width=1.0,
        scalar_bar_args={"title": "|grad_D u - grad_S u|"},
    )
    plotter.add_mesh(arrows, color="black", lighting=True)
    plotter.camera_position = "iso"
    plotter.link_views()
    plotter.enable_anti_aliasing("ssaa")
    plotter.show(screenshot=str(screenshot), auto_close=True)


def main() -> int:
    args = parse_arguments()
    if not math.isfinite(args.radius) or args.radius <= 0.0:
        raise ValueError("--radius doit être strictement positif.")
    if not math.isfinite(args.lc) or args.lc <= 0.0:
        raise ValueError("--lc doit être strictement positif.")
    if not math.isfinite(args.arrow_scale) or args.arrow_scale <= 0.0:
        raise ValueError("--arrow-scale doit être strictement positif.")
    if args.sphere_resolution < 20:
        raise ValueError("--sphere-resolution doit être supérieur ou égal à 20.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    primal_file = args.outdir / "sphere_primal_for_gradient.msh"
    diamond_file = args.outdir / "sphere_diamonds_for_gradient.msh"
    diamond_map_file = args.outdir / "sphere_diamonds_for_gradient.csv"
    csv_file = args.outdir / "sphere_ddfv_gradient.csv"
    screenshot = args.screenshot or args.outdir / "sphere_ddfv_gradient.png"

    gmsh.initialize([])
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        primal_points, triangles = create_primal_mesh(
            args.radius, args.lc, primal_file
        )
        diamonds, face_centers = build_diamonds(
            primal_points, triangles, args.radius
        )
        write_diamond_mesh(diamonds, diamond_file, diamond_map_file)
    finally:
        gmsh.finalize()

    centers, gradients, errors = compute_gradients(
        diamonds, primal_points, face_centers, args.radius
    )
    write_gradient_csv(csv_file, diamonds, centers, gradients, errors)
    render(
        args.radius,
        diamonds,
        centers,
        gradients,
        errors,
        screenshot,
        args.off_screen,
        args.arrow_scale,
        args.sphere_resolution,
    )

    print("Champ : u(x,y,z) = sin(3x) cos(2y) + 0.35 sin(4z)")
    print(f"Diamants             : {len(diamonds)}")
    print(f"Erreur L2 empirique  : {np.sqrt(np.mean(errors**2)):.6e}")
    print(f"Erreur maximale      : {errors.max():.6e}")
    print(f"Valeurs des gradients: {csv_file}")
    print(f"Maillage des diamants: {diamond_file}")
    print(f"Visualisation        : {screenshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python3 sphere_ddfv_gradient_pyvista.py --radius 1 --lc 0.3 --outdir outputs
