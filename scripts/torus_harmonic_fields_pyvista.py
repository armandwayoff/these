#!/usr/bin/env python3
"""Illustre une base des champs harmoniques sur un tore de revolution.

Avec la parametrisation

    X(theta, phi) = ((R + r cos(theta)) cos(phi),
                     (R + r cos(theta)) sin(phi), r sin(theta)),

la metrique induite est ``r^2 dtheta^2 + a(theta)^2 dphi^2``, ou
``a(theta) = R + r cos(theta)``. Les duaux metriques des 1-formes

    dphi,                 dtheta / a(theta)

forment une base de l'espace des champs harmoniques. Le script dessine ces
deux champs, a une constante multiplicative globale pres dans chaque panneau.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--major-radius", type=float, default=1.5)
    parser.add_argument("--minor-radius", type=float, default=0.5)
    parser.add_argument("--surface-resolution", type=int, default=120)
    parser.add_argument(
        "--line-density",
        type=int,
        default=56,
        help="Nombre de lignes de champ dans chaque panneau (défaut : 56).",
    )
    parser.add_argument(
        "--show-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Affiche ou masque les lignes de courant.",
    )
    parser.add_argument(
        "--show-arrows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Affiche ou masque les flèches orientant les lignes.",
    )
    parser.add_argument(
        "--arrows-per-line", type=int, default=3,
        help="Nombre de flèches par ligne sélectionnée (défaut : 3).",
    )
    parser.add_argument(
        "--arrow-scale", type=float, default=0.17,
        help="Longueur des flèches (défaut : 0.17).",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Ancien chemin de base : ajoute les suffixes _longitudinal et _meridional.",
    )
    parser.add_argument(
        "--longitudinal-output",
        type=Path,
        default=None,
        help="Sortie du champ longitudinal (PDF par défaut).",
    )
    parser.add_argument(
        "--meridional-output",
        type=Path,
        default=None,
        help="Sortie du champ méridien (PDF par défaut).",
    )
    parser.add_argument("--off-screen", action="store_true")
    return parser.parse_args()


def torus_mesh(R: float, r: float, resolution: int) -> pv.PolyData:
    theta, phi = np.meshgrid(
        np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False),
        np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False),
        indexing="ij",
    )
    a = R + r * np.cos(theta)
    points = np.column_stack(
        (
            (a * np.cos(phi)).ravel(),
            (a * np.sin(phi)).ravel(),
            (r * np.sin(theta)).ravel(),
        )
    )
    faces = []
    for i in range(resolution):
        for j in range(resolution):
            p = i * resolution + j
            q = ((i + 1) % resolution) * resolution + j
            s = ((i + 1) % resolution) * resolution + (j + 1) % resolution
            t = i * resolution + (j + 1) % resolution
            faces.extend((4, p, q, s, t))
    return pv.PolyData(points, np.asarray(faces))


def field_lines(
    R: float, r: float, density: int, longitudinal: bool, resolution: int = 240
) -> pv.PolyData:
    """Construit les courbes integrales fermees d'un champ de la base."""
    points: list[np.ndarray] = []
    cells: list[int] = []
    parameter = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    for fixed in np.linspace(0.0, 2.0 * np.pi, density, endpoint=False):
        if longitudinal:
            theta = np.full_like(parameter, fixed)
            phi = parameter
        else:
            theta = parameter
            phi = np.full_like(parameter, fixed)
        a = R + r * np.cos(theta)
        curve = np.column_stack(
            (a * np.cos(phi), a * np.sin(phi), r * np.sin(theta))
        )
        offset = len(points)
        points.extend(curve)
        # Le premier indice est repete pour fermer explicitement la polyligne.
        cells.extend((resolution + 1, *range(offset, offset + resolution), offset))
    return pv.PolyData(np.asarray(points), lines=np.asarray(cells))


def directional_arrows(
    R: float,
    r: float,
    line_density: int,
    arrows_per_line: int,
    longitudinal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Place des flèches unitaires tangentes sur une ligne sur deux."""
    points: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    fixed_values = np.linspace(0.0, 2.0 * np.pi, line_density, endpoint=False)[::2]
    moving_values = np.linspace(0.0, 2.0 * np.pi, arrows_per_line, endpoint=False)
    for fixed_index, fixed in enumerate(fixed_values):
        moving = moving_values + fixed_index * np.pi / line_density
        if longitudinal:
            theta = np.full_like(moving, fixed)
            phi = moving
            tangent = np.column_stack(
                (-np.sin(phi), np.cos(phi), np.zeros_like(phi))
            )
        else:
            theta = moving
            phi = np.full_like(moving, fixed)
            tangent = np.column_stack(
                (-np.sin(theta) * np.cos(phi),
                 -np.sin(theta) * np.sin(phi),
                 np.cos(theta))
            )
        a = R + r * np.cos(theta)
        points.extend(
            np.column_stack((a * np.cos(phi), a * np.sin(phi), r * np.sin(theta)))
        )
        vectors.extend(tangent)
    return np.asarray(points), np.asarray(vectors)


def add_output_suffix(path: Path, suffix: str) -> Path:
    """Insère un suffixe avant l'extension d'un chemin."""
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def render(args: argparse.Namespace) -> None:
    if args.major_radius <= args.minor_radius or args.minor_radius <= 0.0:
        raise ValueError("Il faut R > r > 0 pour un tore annulaire regulier.")

    mesh = torus_mesh(
        args.major_radius, args.minor_radius, args.surface_resolution
    )
    if args.line_density < 2:
        raise ValueError("La densite de lignes doit etre au moins egale a 2.")
    if args.arrows_per_line < 1:
        raise ValueError("Il faut au moins une fleche par ligne selectionnee.")
    if not args.show_lines and not args.show_arrows:
        raise ValueError("Les lignes et les fleches ne peuvent pas etre masquees ensemble.")
    h_long = field_lines(
        args.major_radius, args.minor_radius, args.line_density, longitudinal=True
    )
    h_mer = field_lines(
        args.major_radius, args.minor_radius, args.line_density, longitudinal=False
    )
    fields = (
        (h_long, True, r"$h_1=(d\varphi)^\sharp$  — cycle longitudinal", "#1565c0"),
        (
            h_mer,
            False,
            r"$h_2=(d\theta/(R+r\cos\theta))^\sharp$  — cycle méridien",
            'deep_cobalt_violet', # 'cornflower_blue', # "#d84315",
        ),
    )

    longitudinal_output = args.longitudinal_output
    meridional_output = args.meridional_output
    if longitudinal_output is None:
        longitudinal_output = (
            add_output_suffix(args.screenshot, "longitudinal")
            if args.screenshot is not None
            else Path("torus_harmonic_longitudinal.pdf")
        )
    if meridional_output is None:
        meridional_output = (
            add_output_suffix(args.screenshot, "meridional")
            if args.screenshot is not None
            else Path("torus_harmonic_meridional.pdf")
        )

    outputs = (longitudinal_output, meridional_output)
    camera_position = np.asarray((4.0, -4.0, 3.0))
    focal_point = np.zeros(3)
    view_direction = focal_point - camera_position
    view_direction /= np.linalg.norm(view_direction)
    view_up = np.asarray((0.0, 0.0, 1.0))
    camera_right = np.cross(view_direction, view_up)
    camera_right /= np.linalg.norm(camera_right)
    view_up = np.cross(camera_right, view_direction)
    view_up /= np.linalg.norm(view_up)
    camera = [tuple(camera_position), tuple(focal_point), tuple(view_up)]
    raster_extensions = {".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff"}
    graphic_extensions = {".pdf", ".svg", ".eps", ".ps", ".tex"}

    for (lines, longitudinal, title, color), output in zip(fields, outputs):
        plotter = pv.Plotter(
            off_screen=args.off_screen, window_size=(1600, 1600), border=False
        )
        rendered_points = [np.asarray(mesh.points)]
        plotter.set_background("white")
        plotter.add_mesh(
            mesh,
            color="#d9dde3",
            smooth_shading=True,
            specular=0,
            opacity=1,
        )
        if args.show_lines:
            plotter.add_mesh(
                lines,
                color=color,
                line_width=10,
                render_lines_as_tubes=True,
            )
            rendered_points.append(np.asarray(lines.points))
        if args.show_arrows:
            arrow_points, arrow_vectors = directional_arrows(
                args.major_radius, args.minor_radius, args.line_density,
                args.arrows_per_line, longitudinal
            )
            plotter.add_arrows(
                arrow_points, arrow_vectors, mag=args.arrow_scale, color=color
            )
            rendered_points.extend(
                (arrow_points, arrow_points + args.arrow_scale * arrow_vectors)
            )

        # Même recadrage que dans torus_ddfv_gradient_pyvista.py : les
        # dimensions de la fenêtre suivent l'emprise projetée de la géométrie.
        relative_points = np.vstack(rendered_points) - focal_point
        half_width = float(np.max(np.abs(relative_points @ camera_right)))
        half_height = float(np.max(np.abs(relative_points @ view_up)))
        projected_aspect_ratio = half_width / half_height
        maximum_dimension = 1600
        if projected_aspect_ratio >= 1.0:
            window_width = maximum_dimension
            window_height = max(1, round(maximum_dimension / projected_aspect_ratio))
        else:
            window_width = max(1, round(maximum_dimension * projected_aspect_ratio))
            window_height = maximum_dimension
        plotter.window_size = (window_width, window_height)

        plotter.camera_position = camera
        plotter.camera.enable_parallel_projection()
        actual_aspect = window_width / window_height
        plotter.camera.parallel_scale = 1.03 * max(
            half_height, half_width / actual_aspect
        )
        plotter.reset_camera_clipping_range()
        plotter.enable_anti_aliasing("ssaa")

        output.parent.mkdir(parents=True, exist_ok=True)
        extension = output.suffix.lower()
        if extension in graphic_extensions:
            plotter.show(auto_close=False)
            plotter.save_graphic(output, title=title, raster=True)
            plotter.close()
        elif extension in raster_extensions:
            plotter.show(screenshot=str(output), auto_close=True)
        else:
            plotter.close()
            supported = ", ".join(sorted(raster_extensions | graphic_extensions))
            raise ValueError(
                f"Extension de rendu non prise en charge : "
                f"{extension or '(aucune)'}. Extensions acceptées : {supported}"
            )

    print(f"Champ longitudinal : {longitudinal_output}")
    print(f"Champ méridional   : {meridional_output}")


if __name__ == "__main__":
    render(parse_arguments())


"""
.venv/bin/python rapport/scripts/torus_harmonic_fields_pyvista.py \
  --off-screen \
  --screenshot rapport/img/torus_harmonic_fields.png

# Lignes et flèches
.venv/bin/python rapport/scripts/torus_harmonic_fields_pyvista.py --off-screen

# Lignes uniquement
.venv/bin/python rapport/scripts/torus_harmonic_fields_pyvista.py \
  --no-show-arrows --off-screen

# Flèches uniquement
.venv/bin/python rapport/scripts/torus_harmonic_fields_pyvista.py \
  --no-show-lines --off-screen

--arrows-per-line 4 --arrow-scale 0.2

.venv/bin/python rapport/scripts/torus_harmonic_fields_pyvista.py \
  --arrows-per-line 10 \
  --arrow-scale 0.4 \
  --off-screen \
  --longitudinal-output rapport/img/torus_harmonic_longitudinal.pdf \
  --meridional-output rapport/img/torus_harmonic_meridional.pdf
"""
