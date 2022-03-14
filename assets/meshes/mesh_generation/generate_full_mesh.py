from generate import make_pneunet_cavity, make_pneunet_body, make_collision_mesh
from rotate import TRANSLATE
import pygmsh
import numpy as np
import os


def deg_to_rad(a):
    return np.pi * a / 180.


ANGLES = deg_to_rad(np.array([22.5, 157.5, 202.5, 337.5]))
DISK_HEIGHT = 16
DISK_RADIUS = 44


def make_all_cavities(geom):
    cavities = []
    for a in ANGLES:
        cavity = make_pneunet_cavity(geom)
        geom.translate(cavity, TRANSLATE)
        geom.rotate(cavity, (0.0, 0.0, 0.0), a, (0.0, 0.0, 1.0))
        cavities.append(cavity)
    return geom.boolean_union(cavities)


def make_disk(geom):
    # p1 = geom.add_point((DISK_RADIUS, 0, 0))
    # p2 = geom.add_point((-DISK_RADIUS, 0, 0))
    # origin = geom.add_point((0, 0, 0))
    # arc1 = geom.add_circle_arc(p1, origin, p2)
    # arc2 = geom.add_circle_arc(p2, origin, p1)
    # loop = geom.add_curve_loop([arc1, arc2])
    # return geom.extrude(loop, [0, 0, DISK_HEIGHT])[1]
    return geom.add_cylinder([0, 0, 0], [0, 0, DISK_HEIGHT], DISK_RADIUS, mesh_size=4.)


def generate_cavity_meshes(outdir):
    for i, a in enumerate(ANGLES):
        with pygmsh.occ.Geometry() as geom:
            cavity = make_pneunet_cavity(geom)
            geom.translate(cavity, TRANSLATE)
            geom.rotate(cavity, (0.0, 0.0, 0.0), a, (0.0, 0.0, 1.0))
            mesh = geom.generate_mesh(dim=2)
            mesh.write(os.path.join(outdir, f'cavity_{i}.stl'))


def generate(outdir):
    os.makedirs(outdir, exist_ok=True)
    generate_cavity_meshes(outdir)
    with pygmsh.occ.Geometry() as geom:
        cavity = make_all_cavities(geom)
        mesh = geom.generate_mesh(dim=2)
        mesh.write(os.path.join(outdir, "all_cavities.stl"))

        bodies = []
        for a in ANGLES:
            body = make_pneunet_body(geom)
            geom.translate(body, TRANSLATE)
            geom.rotate(body, (0, 0, 0), a, (0, 0, 1))
            bodies.append(body)
        bodies.append(make_disk(geom))
        body = geom.boolean_union(bodies)
        body = geom.boolean_difference(body, cavity)
        mesh = geom.generate_mesh(dim=3)
        mesh.write(os.path.join(outdir, "body.vtk"))

    with pygmsh.occ.Geometry() as geom:
        bodies = []
        for a in ANGLES:
            body = make_collision_mesh(geom)
            geom.translate(body, TRANSLATE)
            geom.rotate(body, (0, 0, 0), a, (0, 0, 1))
            bodies.append(body)
        bodies.append(make_disk(geom))
        body = geom.boolean_union(bodies)
        mesh = geom.generate_mesh(dim=2)
        mesh.write(os.path.join(outdir, "collision.stl"))


if __name__ == '__main__':
    generate('./single_mesh')
