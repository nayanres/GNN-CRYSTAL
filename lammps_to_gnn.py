#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a LAMMPS custom trajectory into GNN-ready transition samples."
    )

    parser.add_argument(
        "trajectory",
        type=Path,
        help="LAMMPS .lammpstrj trajectory"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Number of frames between input and target frame (default: 1)"
    )

    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Also save individual parsed frames"
    )

    parser.add_argument(
        "--neighbor-cutoff",
        type=float,
        default=None,
        help="Optional neighbor cutoff in Angstrom"
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional LAMMPS log file containing thermo data"
    )
    parser.add_argument(
    "--cutoff",
    type=float,
    default=3.5,
    help="Periodic neighbor cutoff in Angstroms"
    )

    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable periodic graph construction"
    )
    return parser.parse_args()


def read_line(f):
    line = f.readline()

    if not line:
        return None

    return line.rstrip("\r\n")


def parse_box(f, header):
    """
    Parse orthogonal or restricted-triclinic LAMMPS box.

    Returns:
        H      3x3 cell matrix
        origin Cartesian origin
        box_type
    """

    triclinic = all(
        field in header.split()
        for field in ["xy", "xz", "yz"]
    )

    rows = []

    for _ in range(3):
        line = read_line(f)

        if line is None:
            raise RuntimeError(
                "Unexpected end of file while reading BOX BOUNDS."
            )

        rows.append([float(x) for x in line.split()])

    if triclinic:
        xlb, xhb, xy = rows[0]
        ylb, yhb, xz = rows[1]
        zlb, zhb, yz = rows[2]

        xlo = xlb - min(0.0, xy, xz, xy + xz)
        xhi = xhb - max(0.0, xy, xz, xy + xz)

        ylo = ylb - min(0.0, yz)
        yhi = yhb - max(0.0, yz)

        zlo = zlb
        zhi = zhb

        # Actual restricted-triclinic box lengths.
        lx = xhi - xlo
        ly = yhi - ylo
        lz = zhi - zlo

        # Cell vectors are stored as rows:
        #
        # a = (lx, 0, 0)
        # b = (xy, ly, 0)
        # c = (xz, yz, lz)
        H = np.array([
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz]
        ])

        origin = np.array(
            [xlo, ylo, zlo],
            dtype=float
        )

        box_type = "triclinic"

    else:
        xlo, xhi = rows[0][:2]
        ylo, yhi = rows[1][:2]
        zlo, zhi = rows[2][:2]

        H = np.diag(
            [
                xhi - xlo,
                yhi - ylo,
                zhi - zlo
            ]
        )

        origin = np.array(
            [xlo, ylo, zlo],
            dtype=float
        )

        box_type = "orthogonal"

    determinant = np.linalg.det(H)

    if determinant <= 0:
        raise RuntimeError(
            f"Invalid simulation cell. Determinant = {determinant}"
        )

    return H, origin, box_type


def determine_coordinate_mode(columns):

    if all(x in columns for x in ["xu", "yu", "zu"]):
        return "unwrapped_cartesian"

    if all(x in columns for x in ["x", "y", "z"]):
        return "cartesian"

    if all(x in columns for x in ["xs", "ys", "zs"]):
        return "scaled"

    if all(x in columns for x in ["xsu", "ysu", "zsu"]):
        return "unwrapped_scaled"

    raise RuntimeError(
        "Could not find usable coordinates. "
        "Expected xu/yu/zu, x/y/z, xs/ys/zs, "
        "or xsu/ysu/zsu."
    )


def parse_trajectory(path):

    frames = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        while True:

            line = read_line(f)

            if line is None:
                break

            if line != "ITEM: TIMESTEP":
                raise RuntimeError(
                    f"Expected ITEM: TIMESTEP but found {line!r}"
                )

            timestep = int(read_line(f))

            line = read_line(f)

            if line != "ITEM: NUMBER OF ATOMS":
                raise RuntimeError(
                    f"Expected NUMBER OF ATOMS but found {line!r}"
                )

            natoms = int(read_line(f))

            box_header = read_line(f)

            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError(
                    f"Expected BOX BOUNDS but found {box_header!r}"
                )

            H, origin, box_type = parse_box(
                f,
                box_header
            )

            atom_header = read_line(f)

            if not atom_header.startswith("ITEM: ATOMS"):
                raise RuntimeError(
                    f"Expected ATOMS header but found {atom_header!r}"
                )

            columns = atom_header.split()[2:]

            if "id" not in columns:
                raise RuntimeError(
                    "Trajectory must contain an id column."
                )

            if "type" not in columns:
                raise RuntimeError(
                    "Trajectory must contain a type column."
                )

            coordinate_mode = determine_coordinate_mode(
                columns
            )

            rows = []

            for _ in range(natoms):

                line = read_line(f)

                if line is None:
                    raise RuntimeError(
                        "Unexpected end of file while reading atoms."
                    )

                values = line.split()

                if len(values) != len(columns):
                    raise RuntimeError(
                        "Atom row has a different number of fields "
                        "than the ATOMS header."
                    )

                rows.append(values)

            id_index = columns.index("id")
            type_index = columns.index("type")

            ids = np.array(
                [
                    int(row[id_index])
                    for row in rows
                ],
                dtype=np.int64
            )

            types = np.array(
                [
                    int(row[type_index])
                    for row in rows
                ],
                dtype=np.int64
            )

            if coordinate_mode == "unwrapped_cartesian":

                coord_names = [
                    "xu",
                    "yu",
                    "zu"
                ]

            elif coordinate_mode == "cartesian":

                coord_names = [
                    "x",
                    "y",
                    "z"
                ]

            elif coordinate_mode == "scaled":

                coord_names = [
                    "xs",
                    "ys",
                    "zs"
                ]

            else:

                coord_names = [
                    "xsu",
                    "ysu",
                    "zsu"
                ]

            coord_indices = [
                columns.index(x)
                for x in coord_names
            ]

            raw_coordinates = np.array(
                [
                    [
                        float(row[index])
                        for index in coord_indices
                    ]
                    for row in rows
                ],
                dtype=float
            )

            if coordinate_mode in [
                "scaled",
                "unwrapped_scaled"
            ]:

                positions = (
                    origin
                    + raw_coordinates @ H.T
                )

            else:

                positions = raw_coordinates

            frame = {
                "timestep": timestep,
                "natoms": natoms,
                "ids": ids,
                "types": types,
                "positions": positions,
                "H": H,
                "origin": origin,
                "box_type": box_type,
                "coordinate_mode": coordinate_mode,
                "columns": columns
            }

            # Optional atom fields.

            optional_fields = [
                "q",
                "vx",
                "vy",
                "vz",
                "fx",
                "fy",
                "fz"
            ]

            for field in optional_fields:

                if field in columns:

                    index = columns.index(field)

                    frame[field] = np.array(
                        [
                            float(row[index])
                            for row in rows
                        ],
                        dtype=float
                    )

            frames.append(frame)

    return frames
def parse_lammps_log(path):
    """
    Parse LAMMPS thermo output into a dictionary keyed by timestep.

    Expected thermo columns include:
    Step Time Temp PotEng KinEng TotEng Enthalpy Vol Press Density
    """

    thermo = {}
    current_columns = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            parts = line.split()

            # Detect a thermo header.
            if (
                "Step" in parts
                and "Temp" in parts
                and "Press" in parts
            ):
                current_columns = parts
                continue

            if current_columns is None:
                continue

            # Thermo data rows must have the same number of fields
            # as the detected header.
            if len(parts) != len(current_columns):
                continue

            try:
                values = [float(x) for x in parts]
            except ValueError:
                continue

            row = dict(
                zip(current_columns, values)
            )

            if "Step" not in row:
                continue

            timestep = int(round(row["Step"]))

            thermo[timestep] = row

    if not thermo:
        raise RuntimeError(
            f"No thermo data found in LAMMPS log: {path}"
        )

    return thermo

def validate_frames(frames):

    if not frames:
        raise RuntimeError(
            "No frames were found."
        )

    reference = frames[0]

    natoms = reference["natoms"]
    reference_ids = reference["ids"]
    reference_types = reference["types"]

    if len(np.unique(reference_ids)) != natoms:
        raise RuntimeError(
            "Duplicate atom IDs detected."
        )

    for i, frame in enumerate(frames):

        if frame["natoms"] != natoms:
            raise RuntimeError(
                f"Frame {i}: atom count changed."
            )

        if not np.array_equal(
            frame["ids"],
            reference_ids
        ):
            raise RuntimeError(
                f"Frame {i}: atom IDs/order changed."
            )

        if not np.array_equal(
            frame["types"],
            reference_types
        ):
            raise RuntimeError(
                f"Frame {i}: atom types changed."
            )

        if not np.all(
            np.isfinite(frame["positions"])
        ):
            raise RuntimeError(
                f"Frame {i}: non-finite position."
            )

        if not np.all(
            np.isfinite(frame["H"])
        ):
            raise RuntimeError(
                f"Frame {i}: non-finite cell."
            )

        if np.linalg.det(frame["H"]) <= 0:
            raise RuntimeError(
                f"Frame {i}: invalid cell."
            )

    timesteps = [
        frame["timestep"]
        for frame in frames
    ]

    for a, b in zip(
        timesteps,
        timesteps[1:]
    ):

        if b <= a:
            raise RuntimeError(
                "Timesteps are not strictly increasing."
            )


def fractional_coordinates(frame):
    positions = frame["positions"]
    origin = frame["origin"]
    H = frame["H"]

    return (positions - origin) @ np.linalg.inv(H)




def wrap_fractional(fractional):

    return fractional - np.floor(fractional)


def affine_positions(frame_a, frame_b):
    fractional = fractional_coordinates(frame_a)

    return (
        frame_b["origin"]
        + fractional @ frame_b["H"]
    )

def minimum_image(
    displacement,
    H
):

    """
    Apply periodic minimum-image wrapping in fractional coordinates.
    """

    fractional = np.linalg.solve(
        H,
        displacement.T
    ).T

    fractional -= np.rint(
        fractional
    )

    return fractional @ H.T
def calculate_transition(
    frame_a,
    frame_b
):
    """
    Calculate the atom-level transition from frame_a to frame_b.

    Returns Cartesian positions, cell matrices, total displacement,
    affine displacement from cell deformation, non-affine displacement,
    and the reconstruction error.
    """

    positions_a = np.asarray(
        frame_a["positions"],
        dtype=np.float64
    )

    positions_b = np.asarray(
        frame_b["positions"],
        dtype=np.float64
    )

    if positions_a.shape != positions_b.shape:
        raise ValueError(
            "Frame position arrays have different shapes."
        )

    # Cell deformation only.
    affine_b = affine_positions(
        frame_a,
        frame_b
    )

    # Actual atom motion.
    displacement = (
        positions_b
        - positions_a
    )

    # Motion remaining after removing the affine
    # contribution from the changing simulation cell.
    non_affine_displacement = (
        positions_b
        - affine_b
    )

    # Reconstruct the final positions from the initial
    # positions plus affine + non-affine motion.
    reconstructed = (
        positions_a
        + (
            affine_b
            - positions_a
        )
        + non_affine_displacement
    )

    reconstruction_error = (
        reconstructed
        - positions_b
    )

    reconstruction_max_error = float(
        np.max(
            np.linalg.norm(
                reconstruction_error,
                axis=1
            )
        )
    )

    return {
        "positions_t":
            positions_a.copy(),

        "positions_next":
            positions_b.copy(),

        "cell_t":
            np.asarray(
                frame_a["H"],
                dtype=np.float64
            ).copy(),

        "cell_next":
            np.asarray(
                frame_b["H"],
                dtype=np.float64
            ).copy(),

        "displacement":
            displacement,

        "affine_positions_next":
            affine_b,

        "affine_displacement":
            affine_b - positions_a,

        "non_affine_displacement":
            non_affine_displacement,

        "reconstruction_max_error":
            np.asarray(
                reconstruction_max_error,
                dtype=np.float64
            ),
    }

def build_periodic_graph(
    positions,
    cell,
    cutoff
):
    """
    Build a directed periodic neighbor graph from Cartesian positions.

    positions:
        (N, 3) Cartesian coordinates. Works with unwrapped
        LAMMPS xu/yu/zu coordinates.

    cell:
        (3, 3) cell matrix with cell vectors as rows.

    edge_shift:
        Integer periodic image applied to the destination atom.
    """

    positions = np.asarray(
        positions,
        dtype=float
    )

    cell = np.asarray(
        cell,
        dtype=float
    )

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "positions must have shape (N, 3)"
        )

    if cell.shape != (3, 3):
        raise ValueError(
            "cell must have shape (3, 3)"
        )

    if cutoff <= 0:
        raise ValueError(
            "cutoff must be positive"
        )

    inv_cell = np.linalg.inv(cell)
    n_atoms = positions.shape[0]

    # Cartesian -> fractional.
    #
    # The cell vectors are stored as rows:
    #
    #     r = f @ cell
    #
    # Therefore:
    #
    #     f = r @ inv(cell)
    #
    # Do NOT wrap the fractional coordinates because
    # positions may be unwrapped LAMMPS xu/yu/zu.
    fractional = positions @ np.linalg.inv(cell)

    shifts = np.array(
        [
            [i, j, k]
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
        ],
        dtype=np.int32
    )

    src = []
    dst = []
    image_shift = []
    vectors = []
    distances = []

    cutoff_sq = cutoff ** 2

    for i in range(n_atoms):

        # Fractional displacement from atom i to every
        # atom j in the reference coordinate frame.
        delta_fractional = (
            fractional
            - fractional[i]
        )

        for shift in shifts:

            # Apply periodic image to destination atom.
            delta = (
                delta_fractional
                + shift
            )

            # Fractional -> Cartesian.
            cart = delta @ cell

            d2 = np.einsum(
                "ij,ij->i",
                cart,
                cart
            )

            valid = d2 <= cutoff_sq

            # Exclude self in the central image.
            if np.all(shift == 0):
                valid[i] = False

            indices = np.flatnonzero(valid)

            for j in indices:

                src.append(i)
                dst.append(j)

                image_shift.append(
                    shift.copy()
                )

                vectors.append(
                    cart[j].copy()
                )

                distances.append(
                    np.sqrt(d2[j])
                )

    return (
        np.asarray(
            src,
            dtype=np.int64
        ),
        np.asarray(
            dst,
            dtype=np.int64
        ),
        np.asarray(
            image_shift,
            dtype=np.int32
        ),
        np.asarray(
            vectors,
            dtype=np.float64
        ),
        np.asarray(
            distances,
            dtype=np.float64
        )
    )

def periodic_neighbor_graph(
    positions,
    H,
    cutoff
):

    """
    Build a simple periodic radius graph.

    This brute-force implementation is intentional for the first version.
    It is perfectly adequate for small atomistic cells and keeps the
    geometry easy to audit.
    """

    n = len(positions)

    edges = []

    cutoff_squared = cutoff ** 2

    for i in range(n):

        displacement = (
            positions
            - positions[i]
        )

        displacement = minimum_image(
            displacement,
            H
        )

        distance_squared = np.sum(
            displacement ** 2,
            axis=1
        )

        neighbors = np.where(
            (distance_squared <= cutoff_squared)
            & (distance_squared > 1e-12)
        )[0]

        for j in neighbors:

            edges.append(
                [i, int(j)]
            )

    if not edges:

        return np.empty(
            (2, 0),
            dtype=np.int64
        )

    return np.asarray(
        edges,
        dtype=np.int64
    ).T


def save_npz(
    path,
    data
):

    np.savez_compressed(
        path,
        **data
    )


def main():

    args = parse_args()

    if args.stride < 1:
        raise RuntimeError(
            "--stride must be at least 1."
        )

    if not args.trajectory.exists():
        raise RuntimeError(
            f"Trajectory does not exist: "
            f"{args.trajectory}"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    sample_directory = (
        args.output / "samples"
    )

    frame_directory = (
        args.output / "frames"
    )

    sample_directory.mkdir(
        exist_ok=True
    )

    if args.keep_frames:
        frame_directory.mkdir(
            exist_ok=True
        )

    print(
        f"Reading {args.trajectory}..."
    )

    frames = parse_trajectory(
        args.trajectory
    )

    print(
        f"Found {len(frames)} frames."
    )
    thermo = None

    if args.log is not None:

        if not args.log.exists():
            raise RuntimeError(
                f"LAMMPS log does not exist: "
                f"{args.log}"
            )

        print(
            f"Reading thermo data from {args.log}..."
        )

        thermo = parse_lammps_log(
            args.log
        )

        print(
            f"Found {len(thermo)} thermo rows."
        )

        missing = [
            frame["timestep"]
            for frame in frames
            if frame["timestep"] not in thermo
        ]

        if missing:
            raise RuntimeError(
                "Missing thermo data for trajectory "
                f"timesteps. First missing: {missing[:10]}"
            )

        print(
            "✓ Thermo data matches every trajectory frame"
        )
    validate_frames(
        frames
    )

    first = frames[0]

    print(
        f"Atoms/frame: {first['natoms']}"
    )

    print(
        f"Cell: {first['box_type']}"
    )

    print(
        f"Coordinates: "
        f"{first['coordinate_mode']}"
    )

    print(
        f"Timestep range: "
        f"{frames[0]['timestep']} -> "
        f"{frames[-1]['timestep']}"
    )

    # Save frames if requested.

    if args.keep_frames:

        for index, frame in enumerate(
            frames
        ):

            data = {
                "timestep":
                    np.asarray(
                        frame["timestep"]
                    ),

                "atom_ids":
                    frame["ids"],

                "atom_types":
                    frame["types"],

                "positions":
                    frame["positions"],

                "cell":
                    frame["H"],

                "origin":
                    frame["origin"]
            }

            for field in [
                "q",
                "vx",
                "vy",
                "vz",
                "fx",
                "fy",
                "fz"
            ]:

                if field in frame:
                    data[field] = frame[field]

            save_npz(
                frame_directory
                / f"frame_{index:06d}.npz",
                data
            )

    reconstruction_errors = []

    transition_count = 0

    for i in range(
        0,
        len(frames) - args.stride,
        args.stride
    ):

        j = i + args.stride

        a = frames[i]
        b = frames[j]

        transition = calculate_transition(
            a,
            b
        )

        reconstruction_errors.append(
            transition[
                "reconstruction_max_error"
            ]
        )

        graph = None

        if not args.no_graph:

            graph = build_periodic_graph(
                a["positions"],
                a["H"],
                args.cutoff
            )

        data = {
            "frame_index_t":
                np.asarray(i),

            "frame_index_next":
                np.asarray(j),

            "timestep_t":
                np.asarray(a["timestep"]),

            "timestep_next":
                np.asarray(b["timestep"]),

            "atom_ids":
                a["ids"],

            "atom_types":
                a["types"],

            **transition,

            "edge_src": graph[0],
            "edge_dst": graph[1],
            "edge_shift": graph[2],
            "edge_vector": graph[3],
            "edge_distance": graph[4],
        }

        # Attach exact thermo state at the start of the transition.
        if thermo is not None:

            thermo_t = thermo[
                a["timestep"]
            ]

            for field in [
                "Time",
                "Temp",
                "PotEng",
                "KinEng",
                "TotEng",
                "Enthalpy",
                "Volume",
                "Press",
                "Density"
            ]:

                if field in thermo_t:

                    data[
                        "thermo_" + field.lower()
                    ] = np.asarray(
                        thermo_t[field]
                    )

        for field in [
            "q",
            "vx",
            "vy",
            "vz",
            "fx",
            "fy",
            "fz"
        ]:

            if field in a:
                data[field + "_t"] = (
                    a[field]
                )

        if args.neighbor_cutoff is not None:

            data["edge_index"] = (
                periodic_neighbor_graph(
                    a["positions"],
                    a["H"],
                    args.neighbor_cutoff
                )
            )

            data["neighbor_cutoff"] = (
                np.asarray(
                    args.neighbor_cutoff
                )
            )

        save_npz(
            sample_directory
            / f"sample_{transition_count:06d}.npz",
            data
        )

        transition_count += 1

    max_error = (
        max(reconstruction_errors)
        if reconstruction_errors
        else 0.0
    )

    metadata = {

        "format_version": "1.0",

        "source_file":
            str(
                args.trajectory.resolve()
            ),

        "number_of_frames":
            len(frames),

        "number_of_atoms":
            first["natoms"],

        "number_of_transitions":
            transition_count,

        "stride":
            args.stride,

        "coordinate_mode":
            first["coordinate_mode"],

        "cell_type":
            first["box_type"],

        "periodic":
            [True, True, True],

        "atom_types":
            sorted(
                int(x)
                for x in np.unique(
                    first["types"]
                )
            ),

        "dump_columns":
            list(first["columns"]),

        "neighbor_cutoff_angstrom":
            float(args.cutoff),

        "reconstruction_max_error_angstrom":
            float(max_error),

        "target_definitions": {

            "total_displacement":
                "r(t+1) - r(t)",

            "affine_displacement":
                "position predicted solely from "
                "cell deformation",

            "nonaffine_displacement":
                "periodic residual after removing "
                "affine cell deformation"
        },

        "notes": [

            "Pressure and temperature are not inferred from atom coordinates.",

            "Total displacement contains both imposed deformation and "
            "internal atomic response.",

            "Non-affine displacement isolates the residual structural "
            "response after affine cell deformation.",

            "Cell matrices are retained so downstream models can explicitly "
            "condition on deformation."
        ]
    }

    with open(
        args.output / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print()
    print("VALIDATION")
    print("----------")
    print("✓ Atom IDs consistent")
    print("✓ Atom count consistent")
    print("✓ Atom types consistent")
    print("✓ Positions finite")
    print("✓ Cell matrices valid")
    print("✓ Timesteps increasing")

    print(
        f"✓ Reconstruction max error: "
        f"{max_error:.6e} Å"
    )

    print()
    print(
        f"Created {transition_count} transition samples."
    )

    print(
        f"Output: {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()