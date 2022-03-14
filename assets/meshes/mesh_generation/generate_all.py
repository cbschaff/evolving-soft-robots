"""Manual rotation of meshes is required to make sure they are exactly the same."""
from generate import generate, generate_disk
from generate import DISK_RADIUS_LOWER, BASE, TOOTH_BASE
import numpy as np
from stl import mesh
import math
from scipy.spatial.transform import Rotation
import os
import meshio

YROT = 15.0  # degrees
y_rad = math.radians(YROT)
z_trans = np.sin(y_rad) * TOOTH_BASE[0] / 2.0
TRANSLATE = (TOOTH_BASE[0] / 2. * np.cos(y_rad) + DISK_RADIUS_LOWER, 0.,
             -BASE[2] + z_trans)
DISK_TRANSLATE = (0.0, 0.0, 2 * z_trans)
ANGLES = [22.5 * i for i in range(16)]  # degrees


def rot_stl(in_file, outfile, angle_y, translation, angle_z):
    quad_colis = in_file
    my_mesh = mesh.Mesh.from_file(quad_colis)
    my_mesh.rotate([0.0, 1.0, 0.0], math.radians(-angle_y))
    my_mesh.translate(translation)
    my_mesh.rotate([0.0, 0.0, 1.0], math.radians(-angle_z))
    my_mesh.save(outfile)


def manual_rotate(in_file, out_file, angle_y, translation, angle_z):
    rotx = Rotation.from_euler('y', angle_y, degrees=True)
    rotz = Rotation.from_euler('z', angle, degrees=True)
    mesh = meshio.read(in_file)
    mesh.points = rotx.apply(mesh.points)
    mesh.points = np.array(translation).reshape(1, 3) + mesh.points
    mesh.points = rotz.apply(mesh.points)
    meshio.gmsh.write(out_file, mesh, "2.2", binary=False)


if __name__ == '__main__':
    generate('./', (0, 0, 0), 0.0, (0, 0, 1))
    generate_disk('../disk')
    for angle in ANGLES:
        angle_str = "{:04d}".format(int(10 * angle))
        print("ANGLE", angle, angle_str)
        os.makedirs(f'../pneunet_{angle_str}', exist_ok=True)
        rot_stl('./cavity.stl', f"../pneunet_{angle_str}/cavity.stl", YROT,
                TRANSLATE, angle)
        rot_stl('./collision.stl', f"../pneunet_{angle_str}/collision.stl",
                YROT, TRANSLATE, angle)
        manual_rotate("./body.vtk", f'../pneunet_{angle_str}/body.msh', YROT,
                      TRANSLATE, angle)

    manual_rotate("../disk/body.vtk", "../disk/body.msh", 0.,
                  DISK_TRANSLATE, 0.)
    rot_stl("../disk/collision.stl", "../disk/collision.stl", 0.,
            DISK_TRANSLATE, 0.)
