# a python script that compute features of .msh file
# read .msh file of tetrahedrons
# compute the total volume of all tetrahedrons
# compute the dimensions of the bounding box of the tetrahedron
# write results in compute_feature_msh.txt
#!/usr/bin/env python3
"""
compute_feature_msh.py

Reads a Gmsh .msh tetrahedral mesh (ASCII, v2.x or v4.x), computes:
- total volume of all tetrahedra
- bounding box min/max and dimensions (dx, dy, dz)

Writes results to compute_feature_msh.txt in the SAME folder as the input .msh.

Notes for fTetWild:
- fTetWild outputs Gmsh .msh (commonly v4.1)
- If the .msh is binary, convert to ASCII using Gmsh first (see error message).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


# ----------------------------
# Geometry
# ----------------------------

def tet_volume(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Volume of a tetrahedron given 4 points."""
    return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


# ----------------------------
# Helpers
# ----------------------------

def _detect_meshformat(lines: List[str]) -> Tuple[float, int]:
    """
    Returns (version, file_type) from $MeshFormat section.
    file_type: 0=ASCII, 1=binary
    """
    for i, line in enumerate(lines):
        if line.strip() == "$MeshFormat":
            parts = lines[i + 1].strip().split()
            version = float(parts[0])
            file_type = int(parts[1]) if len(parts) > 1 else 0
            return version, file_type
    raise ValueError("Could not find $MeshFormat in the .msh file.")


# ----------------------------
# Gmsh v2.x ASCII reader
# ----------------------------

def _read_msh_v2_ascii(lines: List[str]) -> Tuple[Dict[int, np.ndarray], List[List[int]]]:
    nodes: Dict[int, np.ndarray] = {}
    tets: List[List[int]] = []

    i = 0
    n = len(lines)
    while i < n:
        s = lines[i].strip()

        if s == "$Nodes":
            i += 1
            num_nodes = int(lines[i].strip())
            i += 1
            for _ in range(num_nodes):
                parts = lines[i].split()
                nid = int(parts[0])
                x, y, z = map(float, parts[1:4])
                nodes[nid] = np.array([x, y, z], dtype=float)
                i += 1

        elif s == "$Elements":
            i += 1
            num_elems = int(lines[i].strip())
            i += 1
            for _ in range(num_elems):
                parts = lines[i].split()
                elem_type = int(parts[1])
                num_tags = int(parts[2])
                conn = list(map(int, parts[3 + num_tags:]))

                # 4 = tet4, 11 = tet10 (use 4 corner nodes)
                if elem_type in (4, 11) and len(conn) >= 4:
                    tets.append(conn[:4])

                i += 1

        i += 1

    if not nodes:
        raise ValueError("No nodes found in v2 $Nodes section.")
    if not tets:
        raise ValueError("No tetrahedra found (expected element type 4 or 11).")

    return nodes, tets


# ----------------------------
# Gmsh v4.x ASCII reader
# ----------------------------

def _read_msh_v4_ascii(lines: List[str]) -> Tuple[Dict[int, np.ndarray], List[List[int]]]:
    nodes: Dict[int, np.ndarray] = {}
    tets: List[List[int]] = []

    i = 0
    n = len(lines)
    while i < n:
        s = lines[i].strip()

        if s == "$Nodes":
            i += 1
            # numEntityBlocks numNodes minNodeTag maxNodeTag
            nb, nn, *_ = map(int, lines[i].split())
            i += 1

            for _ in range(nb):
                # entityDim entityTag parametric numNodesInBlock
                _, _, _, nblock = map(int, lines[i].split())
                i += 1

                # read node tags (may span multiple lines)
                tags: List[int] = []
                while len(tags) < nblock:
                    tags.extend(map(int, lines[i].split()))
                    i += 1

                # read coordinates: nblock lines with x y z (parametric ignored if present)
                for k in range(nblock):
                    parts = lines[i].split()
                    x, y, z = map(float, parts[:3])
                    nodes[tags[k]] = np.array([x, y, z], dtype=float)
                    i += 1

        elif s == "$Elements":
            i += 1
            # numEntityBlocks numElements minElementTag maxElementTag
            nb, *_ = map(int, lines[i].split())
            i += 1

            for _ in range(nb):
                # entityDim entityTag elementType numElementsInBlock
                _, _, elem_type, nblock = map(int, lines[i].split())
                i += 1

                for _e in range(nblock):
                    parts = lines[i].split()
                    # elementTag followed by node tags
                    conn = list(map(int, parts[1:]))

                    if elem_type in (4, 11) and len(conn) >= 4:
                        tets.append(conn[:4])

                    i += 1

        i += 1

    if not nodes:
        raise ValueError("No nodes found in v4 $Nodes section.")
    if not tets:
        raise ValueError("No tetrahedra found (expected element type 4 or 11).")

    return nodes, tets


def read_msh_ascii(path: Path) -> Tuple[Dict[int, np.ndarray], List[List[int]], float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    version, file_type = _detect_meshformat(lines)
    if file_type != 0:
        raise ValueError(
            "This .msh appears to be BINARY (MeshFormat file_type=1).\n"
            "Convert to ASCII with Gmsh, e.g.:\n"
            "  gmsh your.msh -save -format msh4 -o your_ascii.msh\n"
            "or (older compatibility):\n"
            "  gmsh your.msh -save -format msh2 -o your_ascii.msh\n"
        )

    if version < 4.0:
        nodes, tets = _read_msh_v2_ascii(lines)
    else:
        nodes, tets = _read_msh_v4_ascii(lines)

    return nodes, tets, version


def compute_features(nodes: Dict[int, np.ndarray], tets: List[List[int]]) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, int]:
    # Unique nodes used by tetrahedra
    used_ids = np.unique(np.array(tets, dtype=int).ravel())
    coords = np.vstack([nodes[int(nid)] for nid in used_ids])

    xyz_min = coords.min(axis=0)
    xyz_max = coords.max(axis=0)
    dims = xyz_max - xyz_min

    total_vol = 0.0
    for n0, n1, n2, n3 in tets:
        total_vol += tet_volume(nodes[int(n0)], nodes[int(n1)], nodes[int(n2)], nodes[int(n3)])

    return float(total_vol), xyz_min, xyz_max, dims, int(len(used_ids))


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python compute_feature_msh.py <mesh.msh>")
        return 2

    msh_path = Path(argv[1]).expanduser().resolve()
    nodes, tets, version = read_msh_ascii(msh_path)

    total_vol, xyz_min, xyz_max, dims, n_used = compute_features(nodes, tets)

    out_path = msh_path.with_name(f"mesh_feature_msh_{msh_path.name}.txt")
    out_text = (
        f"msh_file: {msh_path.name}\n"
        f"msh_version: {version}\n"
        f"tetrahedra_count: {len(tets)}\n"
        f"unique_nodes_used_by_tets: {n_used}\n"
        "\n"
        f"total_volume: {total_vol:.12g}\n"
        "\n"
        f"bbox_min_xyz: {xyz_min[0]:.6g}, {xyz_min[1]:.6g}, {xyz_min[2]:.6g}\n"
        f"bbox_max_xyz: {xyz_max[0]:.6g}, {xyz_max[1]:.6g}, {xyz_max[2]:.6g}\n"
        f"bbox_dimensions_dx_dy_dz: {dims[0]:.6g}, {dims[1]:.6g}, {dims[2]:.6g}\n"
    )
    out_path.write_text(out_text, encoding="utf-8")
    print(f"Output written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

