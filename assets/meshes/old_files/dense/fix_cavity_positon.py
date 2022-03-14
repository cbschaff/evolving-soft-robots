import numpy as np
from stl import mesh

# Using an existing stl file:




your_mesh = mesh.Mesh.from_file('cavity_0_0_0.stl')
your_mesh.z += 15
your_mesh.y -= 2
your_mesh.x -= 2
your_mesh.save('new_stl_file.stl')


