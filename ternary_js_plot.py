#!/usr/bin/env python3
"""
ternary_js_plot.py — Same ternary plot as Ehull version, but colorbar = Js (0 → 2.5 by default)

What it does
------------
- Reads a CSV like: index,F,E0,mag,vol,nFe,nCo,nS,Eform,Ehull,Js
  (also accepts E_form/E_hull; Js column required).
- Keeps the lowest-Ehull entry per *reduced* composition (divide by gcd).
- Colors all *non-stable* points by Js using a rainbow colormap (vmin..vmax).
- Stable points (Ehull <= zero_tol) are plotted as larger black dots with white edge.
- Draws bold black outer triangle (Fe top, Co bottom-left, S bottom-right).
- Draws light dashed iso-composition gridlines (0.2, 0.4, 0.6, 0.8 of Fe/Co/S).
- Draws binary polylines along Fe–Co, Co–S, Fe–S using stable points on each edge.
- Builds the *lower* convex hull in (x=fFe, y=fCo, z=Eform) from ALL stable vertices
  (Fe/Co/S + every stable binary/ternary) and draws *all* tie-lines.

Usage
-----
  python ternary_js_plot.py --in processed_data.csv --out ternary_Js.png
  python ternary_js_plot.py --in processed_data.csv --js-min 0.0 --js-max 2.5 --zero-tol 1e-6 --facet-tol 1e-4 --dpi 200
"""

import argparse
import math
from itertools import combinations
from typing import Tuple, List, Dict, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- small helpers -----------------------------

def gcd3(a: int, b: int, c: int) -> int:
    from math import gcd
    return gcd(gcd(abs(a), abs(b)), abs(c))

def reduce_composition(nFe: int, nCo: int, nS: int) -> Tuple[int, int, int]:
    g = gcd3(max(nFe, 0), max(nCo, 0), max(nS, 0))
    g = g if g > 0 else 1
    return (nFe // g, nCo // g, nS // g)

def formula_from_counts(nFe: int, nCo: int, nS: int) -> str:
    parts = []
    for el, n in (("Fe", nFe), ("Co", nCo), ("S", nS)):
        if n <= 0:
            continue
        parts.append(el)
        if n > 1:
            parts.append(r"$_{" + f"{n}" + r"}$")
    return "".join(parts) if parts else "—"

def fractions_from_counts(nFe: int, nCo: int, nS: int) -> Tuple[float, float, float]:
    tot = nFe + nCo + nS
    if tot <= 0:
        return (0.0, 0.0, 0.0)
    return (nFe / tot, nCo / tot, nS / tot)

def barycentric_to_xy(fFe: float, fCo: float, fS: float) -> Tuple[float, float]:
    # Equilateral triangle: Fe at (0.5, sqrt(3)/2), Co at (0,0), S at (1,0)
    h = math.sqrt(3)/2.0
    x = fS + 0.5*fFe
    y = h*fFe
    return x, y

def is_segment_intersect(line1, line2, tol=1e-6):
    """
    line1: ((x1, y1), (x2, y2))
    line2: ((x3, y3), (x4, y4))
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    # --- 1. Bounding Box Check ---
    if max(x1, x2) < min(x3, x4) - tol or \
       max(x3, x4) < min(x1, x2) - tol or \
       max(y1, y2) < min(y3, y4) - tol or \
       max(y3, y4) < min(y1, y2) - tol:
        return False

    # --- 2. Straddle Test ---
    def cross_sign(xa, ya, xb, yb, xc, yc):
        val = (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa)
        if val > tol:
            return 1
        if val < -tol:
            return -1
        return 0
    s1 = cross_sign(x1, y1, x2, y2, x3, y3)
    s2 = cross_sign(x1, y1, x2, y2, x4, y4)
    s3 = cross_sign(x3, y3, x4, y4, x1, y1)
    s4 = cross_sign(x3, y3, x4, y4, x2, y2)
    if (s1 * s2 <= 0) and (s3 * s4 <= 0):
        return True

    return False

def is_line_box_intersect(line, box, edge_clearance, tol=1e-6):
    """
    line: ((lx1, ly1), (lx2, ly2))
    box:  (x0, y0, x1, y1)
    """
    (lx1, ly1), (lx2, ly2) = line
    bx0, by0, bx1, by1 = box

    bx0 -= edge_clearance
    by0 -= edge_clearance
    bx1 += edge_clearance
    by1 += edge_clearance

    # --- 1. Endpoint Inclusion Test ---
    def is_inside(x, y):
        return (bx0 - tol <= x <= bx1 + tol) and \
               (by0 - tol <= y <= by1 + tol)
    if is_inside(lx1, ly1) or is_inside(lx2, ly2):
        return True

    # --- 2. Diagonal Intersection Test ---
    diag1 = ((bx0, by0), (bx1, by1))
    diag2 = ((bx0, by1), (bx1, by0))
    if is_segment_intersect(line, diag1, tol) or \
       is_segment_intersect(line, diag2, tol):
        return True

    return False

def is_box_intersect(box1, box2, edge_clearance, tol=1e-6):
    """
    box1: (ax0, ay0, ax1, ay1)
    box2: (bx0, by0, bx1, by1)
    """
    (ax0, ay0, ax1, ay1) = box1
    (bx0, by0, bx1, by1) = box2

    # --- Separating Axis Theorem ---
    if ax1 < bx0 - edge_clearance - tol:
        return False
    if ax0 > bx1 + edge_clearance + tol:
        return False
    if ay1 < by0 - edge_clearance - tol:
        return False
    if ay0 > by1 + edge_clearance + tol:
        return False

    return True

def triangle_vertices():
    h = math.sqrt(3)/2.0
    Fe = (0.5, h); Co = (0.0, 0.0); S = (1.0, 0.0)
    return Fe, Co, S, h


# ----------------------------- drawing bits -----------------------------

def plot_triangle(ax):
    Fe, Co, S, h = triangle_vertices()
    ax.plot([Co[0], S[0]],[Co[1], S[1]], color='#222', lw=1.6, zorder=1)
    ax.plot([S[0], Fe[0]],[S[1], Fe[1]], color='#222', lw=1.6, zorder=1)
    ax.plot([Fe[0], Co[0]],[Fe[1], Co[1]], color='#222', lw=1.6, zorder=1)
    ax.text(Fe[0], Fe[1]+0.04, "Fe", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(Co[0]-0.04, Co[1]-0.03, "Co", ha="right", va="top", fontsize=11, fontweight="bold")
    ax.text(S[0]+0.04,  S[1]-0.03, "S",  ha="left",  va="top", fontsize=11, fontweight="bold")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, h+0.08)
    ax.axis('off')
    return [(Co, S), (S, Fe), (Fe, Co)]

def draw_gridlines(ax, steps=(0.2, 0.4, 0.6, 0.8)):
    segs = []
    def add_line(p1, p2):
        ax.plot([p1[0], p2[0]],[p1[1], p2[1]], 'k--', lw=0.5, alpha=0.25, zorder=0.5)
        segs.append((p1, p2))
    for t in steps:
        # const Fe = t
        add_line(barycentric_to_xy(t, 1-t, 0), barycentric_to_xy(t, 0, 1-t))
        # const Co = t
        add_line(barycentric_to_xy(1-t, t, 0), barycentric_to_xy(0, t, 1-t))
        # const S  = t
        add_line(barycentric_to_xy(1-t, 0, t), barycentric_to_xy(0, 1-t, t))
    return segs

def sort_and_connect(points, ax, **kwargs):
    if not points: return []
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    segs = []
    for i in range(len(pts)-1):
        p1, p2 = pts[i], pts[i+1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)
        segs.append((p1, p2))
    return segs

def choose_label_position(x, y, taken_boxes, text, ax, renderer, avoid_lines, edge_clearance=0.022):
    r_list = np.arange(0.02, 0.07, 0.005)
    angles = np.linspace(0, 2*np.pi, 48, endpoint=False)
    def get_text_bbox(xp, yp):
        t = ax.text(xp, yp, text, fontsize=8, ha="center", va="center", color="black",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.2),
                    zorder=5)
        plt.draw()
        bb = t.get_window_extent(renderer=renderer)
        inv = ax.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform(bb)
        t.remove()
        return (x0, y0, x1, y1)
    for r in r_list:
        for ang in angles:
            dx, dy = r*math.cos(ang), r*math.sin(ang)
            xp, yp = x+dx, y+dy
            current_box = get_text_bbox(xp, yp)
            if any(is_box_intersect(current_box, tb, edge_clearance) for tb in taken_boxes):
                continue
            if any(is_line_box_intersect(line, current_box, edge_clearance) for line in avoid_lines):
                continue
            return xp, yp
    return x, y+0.03


# ----------------------------- lower-hull (tie-lines) -----------------------------

def fit_plane(p1, p2, p3, tol_area=1e-12):
    # p = (x,y,z). Return (a,b,c) for z = a x + b y + c, or None if nearly collinear.
    (x1,y1,z1), (x2,y2,z2), (x3,y3,z3) = p1, p2, p3
    area2 = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    if area2 < tol_area:
        return None
    A = np.array([[x1,y1,1.0],[x2,y2,1.0],[x3,y3,1.0]], dtype=float)
    b = np.array([z1,z2,z3], dtype=float)
    a,beta,c = np.linalg.solve(A,b)
    return a, beta, c

def lower_hull_edges_xy(stable_xyz: List[Tuple[float,float,float]], tol: float = 1e-4) -> Set[Tuple[Tuple[float,float],Tuple[float,float]]]:
    """
    stable_xyz: list of (x, y, z=Eform) for STABLE vertices (elements + binaries + ternaries).
    Returns undirected edges ( (x1,y1),(x2,y2) ) on the LOWER convex hull graph.
    'tol' is the hull inequality tolerance (default 1e-4).
    """
    N = len(stable_xyz)
    edges: Set[Tuple[Tuple[float,float],Tuple[float,float]]] = set()
    if N < 3:
        return edges
    idxs = list(range(N))
    for i, j, k in combinations(idxs, 3):
        p1, p2, p3 = stable_xyz[i], stable_xyz[j], stable_xyz[k]
        plane = fit_plane(p1, p2, p3)
        if plane is None:
            continue
        a, b, c = plane
        # Keep if plane is below (or equal within tol) every stable vertex:
        # z >= a x + b y + c - tol  (lower hull)
        for (x,y,z) in stable_xyz:
            if z < (a*x + b*y + c) - tol:
                break
        else:
            xy1 = (p1[0], p1[1]); xy2 = (p2[0], p2[1]); xy3 = (p3[0], p3[1])
            for u,v in ((xy1,xy2),(xy2,xy3),(xy3,xy1)):
                if u != v:
                    edges.add(tuple(sorted((u,v))))
    return edges


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Ternary (Fe top, Co BL, S BR): stable black; others rainbow by Js; binary + ternary tie-lines; grid.")
    ap.add_argument("--in", dest="input_csv", required=True, help="processed_data.csv")
    ap.add_argument("--out", dest="output_png", default="ternary_Js.png", help="Output PNG")
    ap.add_argument("--js-min", type=float, default=0.0, help="Colorbar lower bound for Js")
    ap.add_argument("--js-max", type=float, default=2.5, help="Colorbar upper bound for Js")
    ap.add_argument("--zero-tol", type=float, default=1e-6, help="Stability tolerance for Ehull≈0")
    ap.add_argument("--facet-tol", type=float, default=1e-4, help="Lower-hull inequality tolerance (z ≥ ax+by+c - tol)")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    # Accept both styles of column names for formation/hull; Js must be present
    col_eh = "E_hull" if "E_hull" in df.columns else ("Ehull" if "Ehull" in df.columns else None)
    col_ef = "E_form" if "E_form" in df.columns else ("Eform" if "Eform" in df.columns else None)
    col_js = "Js" if "Js" in df.columns else ("J_s" if "J_s" in df.columns else None)
    if col_eh is None or col_ef is None or col_js is None:
        raise ValueError("CSV must contain Js (or J_s) and Ehull/E_hull and Eform/E_form, plus nFe,nCo,nS.")

    for col in ("nFe","nCo","nS"):
        if col not in df.columns:
            raise ValueError("CSV must contain integer columns: nFe, nCo, nS")

    # Keep lowest-Ehull per reduced composition; carry Js from that record
    best: Dict[Tuple[int,int,int], Dict] = {}
    for _, r in df.iterrows():
        nFe, nCo, nS = int(r["nFe"]), int(r["nCo"]), int(r["nS"])
        key = reduce_composition(nFe, nCo, nS)
        eh  = float(r[col_eh])
        if key not in best or eh < best[key][col_eh]:
            best[key] = {
                "nFe": key[0], "nCo": key[1], "nS": key[2],
                col_eh: eh,
                col_ef: float(r[col_ef]),
                col_js: float(r[col_js]) if pd.notnull(r[col_js]) else np.nan,
                "formula": r.get("formula", formula_from_counts(*key))
            }

    # Prepare points & stable sets
    pts_js, stable_pts = [], []
    stable_xyz_all: List[Tuple[float,float,float]] = []  # for lower hull

    # Add ELEMENT vertices explicitly (Eform = 0)
    Fe_v, Co_v, S_v, _ = triangle_vertices()
    stable_xyz_all.extend([(Fe_v[0], Fe_v[1], 0.0),
                           (Co_v[0], Co_v[1], 0.0),
                           (S_v[0],  S_v[1],  0.0)])

    for key, rec in best.items():
        nFe, nCo, nS = rec["nFe"], rec["nCo"], rec["nS"]
        fFe, fCo, fS = fractions_from_counts(nFe, nCo, nS)
        x, y = barycentric_to_xy(fFe, fCo, fS)
        eh   = float(rec[col_eh])
        ef   = float(rec[col_ef])
        js   = float(rec[col_js]) if np.isfinite(rec[col_js]) else np.nan
        pts_js.append((x, y, js, eh))
        if abs(eh) <= args.zero_tol:
            stable_pts.append((x, y, rec["formula"], nFe, nCo, nS))
            stable_xyz_all.append((x, y, ef))

    if not pts_js:
        raise RuntimeError("No points to plot after grouping by composition.")

    fig, ax = plt.subplots(figsize=(7.4, 6.9), dpi=args.dpi)
    triangle_edges = plot_triangle(ax)
    grid_segments  = draw_gridlines(ax, steps=(0.2, 0.4, 0.6, 0.8))

    # Scatter non-stable points colored by Js (skip NaNs)
    arr = np.array(pts_js, dtype=float)
    x_all, y_all, js_all, eh_all = arr[:,0], arr[:,1], arr[:,2], arr[:,3]
    mask_plot = (eh_all > args.zero_tol) & np.isfinite(js_all)
    if np.any(mask_plot):
        js = np.clip(js_all[mask_plot], args.js_min, args.js_max)
        sc = ax.scatter(x_all[mask_plot], y_all[mask_plot],
                        c=js, s=28, cmap='rainbow', vmin=args.js_min, vmax=args.js_max,
                        edgecolors='none', zorder=2)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$J_\mathrm{s}$ (T)")

    # Stable points (bigger, with white edge so they pop)
    if len(stable_pts):
        xs = [p[0] for p in stable_pts]; ys = [p[1] for p in stable_pts]
        ax.scatter(xs, ys, s=80, c='black', edgecolors='white', linewidths=0.7, zorder=4)

    # Binary hull polylines on triangle edges (stable points on each edge)
    FeCo_edge, CoS_edge, FeS_edge = [], [], []
    for (x, y, _, nFe, nCo, nS) in stable_pts:
        if nS == 0: FeCo_edge.append((x,y))
        if nFe == 0: CoS_edge.append((x,y))
        if nCo == 0: FeS_edge.append((x,y))
    # include corners
    for p, edge in ((Fe_v,FeCo_edge),(Co_v,FeCo_edge),(Co_v,CoS_edge),(S_v,CoS_edge),(S_v,FeS_edge),(Fe_v,FeS_edge)):
        if p not in edge: edge.append(p)

    binary_segments = []
    binary_segments += sort_and_connect(FeCo_edge, ax, color='k', lw=1.6, zorder=2.6)
    binary_segments += sort_and_connect(CoS_edge, ax, color='k', lw=1.6, zorder=2.6)
    binary_segments += sort_and_connect(FeS_edge, ax, color='k', lw=1.6, zorder=2.6)

    # Full Gibbs network from LOWER convex hull across ALL stable vertices (elements+binaries+ternaries)
    ternary_segments = []
    if len(stable_xyz_all) >= 3:
        edges = lower_hull_edges_xy(stable_xyz_all, tol=args.facet_tol)
        for (p, q) in edges:
            (x1,y1), (x2,y2) = p, q
            ax.plot([x1,x2],[y1,y2], color='k', lw=1.5, alpha=0.95, zorder=3.1)
            ternary_segments.append((p,q))

    # Labels (avoid all lines)
    if len(stable_pts):
        renderer = fig.canvas.get_renderer()
        placed: List[Tuple[float,float,float,float]] = []
        avoid_lines = triangle_edges + binary_segments + ternary_segments + grid_segments
        for (x, y, text, _, _, _) in stable_pts:
            if text in ["Fe", "Co", "S"]:
                continue
            xp, yp = choose_label_position(x, y, placed, text, ax, renderer, avoid_lines, edge_clearance=0.01)
            t = ax.text(xp, yp, text, fontsize=8, ha="center", va="center", color="black",
                        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.2), zorder=5)
            fig.canvas.draw()
            (x0,y0),(x1,y1) = ax.transData.inverted().transform(t.get_window_extent(renderer=renderer))
            placed.append((x0,y0,x1,y1))

    #ax.set_title("Fe–Co–S Ternary: stable (black), full Gibbs tie-lines, J_s (rainbow)", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=args.dpi)
    print(f"Saved {args.output_png}")


if __name__ == "__main__":
    main()

