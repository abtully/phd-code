"""
Generates and plots polygons
@author: Alexandra Tully
@date: March 9, 2021

1. generate polygon with scale, rotation (in degrees), and translation; returns list of coordinates
2. plot coordinates with lines
"""
from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

DEFAULT_RENDERER = 'browser'  # this is a constant of this file


def gen_polygon(sides: int, radius: float = 1, rotation: float = 0, translation: Tuple[float, float] = None) -> \
        List[Tuple]:
    """
    Function source: https://stackoverflow.com/questions/23411688/drawing-polygon-with-n-number-of-sides-in-python-3-2?fbclid=IwAR0_TdiNKMnbGR3NCDX3xKDV-dtNjaQxZgNZ4hPkA3t_n9kr_84UtA7kJ_s
    "The basic approach is to sweep out a circle, and put n equally spaced points on it. These will be the points of our
    polygon, starting at the 12 'o' clock position."

    Args:
        sides: number of sides for polygon
        radius: scales polygon
        rotation: rotation in degrees
        translation: translates polygon by (x, y)

    Returns: List of coordinates (tuples)

    """
    one_segment = np.pi * 2 / sides
    rotation = rotation * np.pi / 180

    points = [
        (np.sin(one_segment * i + rotation) * radius,
         np.cos(one_segment * i + rotation) * radius)
        for i in range(sides)]

    if translation:
        points = [[sum(pair) for pair in zip(point, translation)]
                  for point in points]

    return points


def center_point(translation: Tuple[float, float] = None) -> tuple:
    center = (0, 0)
    if translation is not None:
        center = (center[0] + translation[0], center[1] + translation[1])
    return center


def gen_hexagon_basis_vectors(coords: List[Tuple] = None, sides: int = None, radius: float = 1, rotation: float = 0,
                              translation: Tuple[float, float] = None) -> List[Tuple]:
    """Either provide list of coordinates and any translation, or all of the parameters needed to generate the base
    hexagon"""
    if coords is None:
        coords = gen_polygon(sides, radius, rotation, translation)
    if len(coords) != 6:
        raise ValueError(f'{len(coords)} is not 6, so this is not a hexagon')
    center = center_point(translation)
    a1 = (coords[0][0] - center[0], coords[0][1] - center[1])
    a2 = (coords[1][0] - center[0], coords[1][1] - center[1])
    return a1, a2, center


def gen_tiled_hexagons(coords: List[Tuple] = None, sides: int = 6, radius: float = 1, rotation: float = 0,
                              translation: Tuple[float, float] = None) -> List[Tuple]:
    if coords is None:
        coords = gen_polygon(sides, radius, rotation, translation)
    if len(coords) != 6:
        raise ValueError(f'{len(coords)} is not 6, so this is not a hexagon')
    a1, a2, center = gen_hexagon_basis_vectors(coords=coords, translation=translation)
    # a_new = tuple(2 * a for a in a2)
    # a_new = tuple([np.mean([a1[0], a2[0]]), np.mean([a1[1], a2[1]])])
    # points = gen_polygon(sides=6, radius=radius, rotation=rotation, translation=a_new)

    points = gen_polygon(sides=6, radius=radius, rotation=rotation, translation=translation)
    points_tr = [[sum(pair) for pair in zip(point, a1)]
              for point in points]
    points_tr = [[sum(pair) for pair in zip(point, a2)]
              for point in points_tr]
    a1_bl, a2_bl = -1 * np.array(a1), -1 * np.array(a2)
    points_bl = [[sum(pair) for pair in zip(point, a1_bl)]
              for point in points]
    points_bl = [[sum(pair) for pair in zip(point, a2_bl)]
              for point in points_bl]
    a1_br, a2_br = -2 * np.array(a1), np.array(a2)
    points_br = [[sum(pair) for pair in zip(point, a1_br)]
              for point in points]
    points_br = [[sum(pair) for pair in zip(point, a2_br)]
              for point in points_br]
    a1_l, a2_l = np.array(a1), -2 * np.array(a2)
    points_l = [[sum(pair) for pair in zip(point, a1_l)]
              for point in points]
    points_l = [[sum(pair) for pair in zip(point, a2_l)]
              for point in points_l]
    a1_tl, a2_tl = 2 * np.array(a1), -1 * np.array(a2)
    points_tl = [[sum(pair) for pair in zip(point, a1_tl)]
              for point in points]
    points_tl = [[sum(pair) for pair in zip(point, a2_tl)]
              for point in points_tl]
    a1_r, a2_r = -1 * np.array(a1), 2 * np.array(a2)
    points_r = [[sum(pair) for pair in zip(point, a1_r)]
              for point in points]
    points_r = [[sum(pair) for pair in zip(point, a2_r)]
              for point in points_r]
    return points_tr, points_bl, points_br, points_l, points_tl, points_r

# def gen_tiled_hexagons(tiles: int, coords: List[Tuple], sides: int = None, radius: float = 1, rotation: float = 0,
#                               translation: Tuple[float, float] = None) -> List[Tuple]:
#     if coords is None:
#         coords = gen_polygon(sides, radius, rotation, translation)
#     if len(coords) != 6:
#         raise ValueError(f'{len(coords)} is not 6, so this is not a hexagon')
#     a1, a2, center = gen_hexagon_basis_vectors(coords, translation)
#     tiles = np.arange(0, tiles, 1)
#     t_coords_a1, t_coords_a2 = [t * np.array(a1) for t in tiles], [t * np.array(a2) for t in tiles]
#     a1_hexs, a2_hexs = [a * np.array(coords) for a in t_coords_a1], [a * np.array(coords) for a in t_coords_a2]
#     return a1_hexs, a2_hexs


def plot_polygon(coords: List[Tuple], color: str = None, darkmode=False, fig=None, show=True, dash=False, dash_width=1) -> go.Figure:
    # if type(coords) == list:
    if fig is None:
        fig = go.Figure()
    endpoint = coords[0]
    coords = coords + [endpoint]
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    if dash:
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color, width=dash_width, dash="dash")))
    else:
        fig.add_trace(go.Scatter(mode='lines', x=x, y=y, marker=dict(color=color)))
    if darkmode:
        fig.update_layout(template='plotly_dark')
    if show:
        fig.show(renderer=DEFAULT_RENDERER)
    return fig


def plot_lattice_vectors(a1: tuple, a2: tuple, center: tuple, coords: List[Tuple], color: str = None, darkmode=False) -> go.Figure:
    fig = plot_polygon(coords=coords, color=color, darkmode=darkmode)
    fig.add_trace(go.Scatter(mode='markers', x=[center[0]], y=[center[1]]))
    x1, y1 = [center[0], center[0] + a1[0]], [center[1], center[1] + a1[1]]
    x2, y2 = [center[0], center[0] + a2[0]], [center[1], center[1] + a2[1]]
    fig.add_trace(go.Scatter(mode='lines', x=x1, y=y1))
    fig.add_trace(go.Scatter(mode='lines', x=x2, y=y2))
    fig.show(renderer=DEFAULT_RENDERER)
    return fig


def plot_polygons(coords: List[List[Tuple]], color: str = None, darkmode=False, fig=None, show=True) -> go.Figure:
    # if type(coords) == list:
    if fig is None:
        fig = go.Figure()
    for i in coords:
        endpoint = i[0]
        i = i + [endpoint]
        x = [c[0] for c in i]
        y = [c[1] for c in i]
        fig.add_trace(go.Scatter(mode='lines', x=x, y=y, marker=dict(color=color)))
    if darkmode:
        fig.update_layout(template='plotly_dark')
    if show:
        fig.show(renderer=DEFAULT_RENDERER)
    return fig


def plot_polygons_mpl(fig=None, ax=None, color='black', material='C60', nsides=6, multiple=False, radius=False, rotation=30, translation=(0, 0), title=None):
    # if type(coords) == list:
    if fig is None:
        fig, ax = plt.subplots(1)
    if not radius:
        if material == 'C60':
            radius = 0.42
        elif material == 'Au(111)':
            radius = 1.1
        else:
            raise ValueError(f'Material {material} not C60 or Au(111), must specify radius argument.')
    coords = gen_polygon(nsides, radius=radius, rotation=rotation, translation=translation)
    if multiple:
        coords, coords_bl, coords_tr, coords_l, coords_tl, coords_r = gen_tiled_hexagons(coords, radius=radius, rotation=rotation, translation=translation)
        new_coords = [coords, coords_bl, coords_tr, coords_l, coords_tl, coords_r]
#         new_coords = [coords, coords_bl, coords_tr, coords_l, coords_r]
        plt_polygons(new_coords, fig=fig, ax=ax, color=color, title=title)
    if not multiple:
        plt_polygon(coords, fig=fig, ax=ax, color=color, title=title)
    return fig


def plt_polygon(coords, fig=None, ax=None, color='black', title=None):
    if fig is None:
        fig, ax = plt.subplots(1)
    endpoint = coords[0]
    coords = coords + [endpoint]
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    ax.plot(x, y, color=color)
    if title is None:
        ax.set_title(label=f'BZ Overlay')
    return fig


def plt_polygons(coords, fig=None, ax=None, color='black', title=None):
    if ax:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    if fig is None:
        fig, ax = plt.subplots(1)
    for i in coords:
        endpoint = i[0]
        i = i + [endpoint]
        x = [c[0] for c in i]
        y = [c[1] for c in i]
        ax.plot(x, y, color=color)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
    if title is None:
        ax.set_title(label=f'BZ Overlay')
    return fig


if __name__ == '__main__':
    # coords = gen_polygon(6, rotation=30, translation=(1, 1))
    # coords = [(1, 2), (2, 1), (2, 0), (0, 1)]
    # plot_polygon(coords, darkmode=True)
    coords = gen_polygon(6)
    new_coords, coords_bl, coords_tr, coords_l, coords_tl, coords_r = gen_tiled_hexagons(coords)
    plot_polygons([coords, new_coords, coords_bl, coords_tr, coords_l, coords_tl, coords_r])
    # a1, a2, center = gen_hexagon_basis_vectors(coords, translation=(1, 1))
    # a1_hexs, a2_hexs = gen_tiled_hexagons(3, coords)
    # endpoint = a1_hexs[1][0]
    # coords_new = np.ndarray.tolist(a1_hexs[1]) + [np.ndarray.tolist(endpoint)]
    # x = [c[0] for c in coords_new]
    # y = [c[1] for c in coords_new]
    # fig = go.Figure(go.Scatter(mode='lines', x=x, y=y, marker=dict(color=None)))
    # fig.update_layout(template='plotly_dark')
    # fig.show(renderer=DEFAULT_RENDERER)

    # plot_lattice_vectors(a1, a2, center, coords, color='red', darkmode=True)

    # fig = plot_polygon(coords=coords, darkmode=True)
    # fig.add_trace(go.Scatter(mode='markers', x=[center[0]], y=[center[1]]))
    # x1, y1 = [center[0], center[0] + a1[0]], [center[1], center[1] + a1[1]]
    # x2, y2 = [center[0], center[0] + a2[0]], [center[1], center[1] + a2[1]]
    # fig.add_trace(go.Scatter(mode='lines', x=x1, y=y1))
    # fig.add_trace(go.Scatter(mode='lines', x=x2, y=y2))
    # x_tiles = [t[0] for t in a2_hexs[1]]
    # y_tiles = [t[1] for t in a2_hexs[1]]
    # fig.add_trace(go.Scatter(mode='markers', x=x_tiles, y=y_tiles))
    # fig.show(renderer=DEFAULT_RENDERER)

    # import quantipy.lattice as latt
    # latt.get_finite_lattices(lattice_type='Honeycomb', n_sites='5')