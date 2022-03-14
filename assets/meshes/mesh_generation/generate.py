import pygmsh
import numpy as np
import os


BASE = [115, 32, 4]
TOOTH_BASE = [104, 20, 9]
BIG_TOOTH = [11, 20, 7]
SMALL_TOOTH = [8, 20, 7]
TOOTH_GAP = 1
BASE_START = [-0.5 * BASE[0], -0.5 * BASE[1], 0.]
TOOTH_BASE_START = [-0.5 * TOOTH_BASE[0], -0.5 * TOOTH_BASE[1], BASE[2]]


CAVITY_CONNECTOR = [81, 2, 2]
CAVITY_TOOTH = [2, 14, 13]
CAVITY_CONNECTOR_START = [-0.5 * CAVITY_CONNECTOR[0],
                          -0.5 * CAVITY_CONNECTOR[1], BASE[2]]
CAVITY_GAP = 7
CAVITY_START = 3.5

COLLISION_TEETH = [TOOTH_BASE[0], TOOTH_BASE[1], TOOTH_BASE[2] + BIG_TOOTH[2]]

DISK_RADIUS_UPPER = 54.5
DISK_RADIUS_LOWER = 50.0


def make_disk(geom, mesh_size):
    geom.add_cone(center=(0, 0, 0), axis=(0, 0, 18),
                  radius0=DISK_RADIUS_LOWER, radius1=DISK_RADIUS_UPPER,
                  mesh_size=mesh_size)


def make_box(geom, dims, start, mesh_size=4.0):
    dims = np.asarray(dims)
    start = np.asarray(start)
    points = [
        start,
        start + dims * np.array([1, 0, 0]),
        start + dims * np.array([1, 1, 0]),
        start + dims * np.array([0, 1, 0]),
    ]
    poly = geom.add_polygon(points, mesh_size=mesh_size)
    return geom.extrude(poly, [0, 0, dims[2]])[1]


def make_pneunet_body(geom):
    x = TOOTH_BASE_START[0]
    parts = [
        make_box(geom, BASE, BASE_START),
        make_box(geom, TOOTH_BASE, TOOTH_BASE_START),
    ]

    if BIG_TOOTH[2] == 0:
        return geom.boolean_union(parts)[0]

    parts.append(
        make_box(geom, BIG_TOOTH,
                 [x, TOOTH_BASE_START[1], BASE[2] + TOOTH_BASE[2]])
    )
    x += BIG_TOOTH[0] + TOOTH_GAP
    for _ in range(9):
        parts.append(
            make_box(geom, SMALL_TOOTH,
                     [x, TOOTH_BASE_START[1], BASE[2] + TOOTH_BASE[2]])
        )
        x += SMALL_TOOTH[0] + TOOTH_GAP
    parts.append(
        make_box(geom, BIG_TOOTH,
                 [x, TOOTH_BASE_START[1], BASE[2] + TOOTH_BASE[2]])
    )
    return geom.boolean_union(parts)[0]


def make_pneunet_cavity(geom):
    parts = []
    parts.append(make_box(geom, CAVITY_CONNECTOR, CAVITY_CONNECTOR_START))
    x = CAVITY_CONNECTOR_START[0] + CAVITY_START
    for _ in range(9):
        parts.append(
            make_box(geom, CAVITY_TOOTH,
                     [x, -0.5 * CAVITY_TOOTH[1], CAVITY_CONNECTOR_START[2]])
        )
        x += CAVITY_TOOTH[0] + CAVITY_GAP
    return geom.boolean_union(parts)[0]


def make_collision_mesh(geom):
    parts = []
    parts.append(make_box(geom, BASE, BASE_START, 10.0))
    parts.append(make_box(geom, COLLISION_TEETH, TOOTH_BASE_START, 10.0))
    return geom.boolean_union(parts)[0]


def generate(outdir, translate=(0, 0, 0), angle=0., axis=(0, 0, 1.0)):
    os.makedirs(outdir, exist_ok=True)
    with pygmsh.occ.Geometry() as geom:
        cavity = make_pneunet_cavity(geom)
        geom.translate(cavity, translate)
        geom.rotate(cavity, (0, 0, 0), angle, axis)
        mesh = geom.generate_mesh(dim=2)
        mesh.write(os.path.join(outdir, "cavity.stl"))

        body = make_pneunet_body(geom)
        geom.translate(body, translate)
        geom.rotate(body, (0, 0, 0), angle, axis)
        body = geom.boolean_difference(body, cavity)
        mesh = geom.generate_mesh(dim=3)
        mesh.write(os.path.join(outdir, "body.vtk"))

    with pygmsh.occ.Geometry() as geom:
        collision = make_collision_mesh(geom)
        geom.translate(collision, translate)
        geom.rotate(collision, (0, 0, 0), angle, axis)
        mesh = geom.generate_mesh(dim=2)
        mesh.write(os.path.join(outdir, "collision.stl"))


def generate_disk(outdir, mesh_size=20):
    with pygmsh.occ.Geometry() as geom:
        make_disk(geom, mesh_size)
        mesh = geom.generate_mesh(dim=3)
        mesh.write(os.path.join(outdir, "body.vtk"))
        mesh = geom.generate_mesh(dim=2)
        mesh.write(os.path.join(outdir, "collision.stl"))


if __name__ == '__main__':
    # generate('./', (0, 0, 0), 0., (0, 0, 1))
    generate_disk('./')
