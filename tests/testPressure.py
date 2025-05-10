import torch
import os
import numpy as np
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import FEA

current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cpu'))
torch.set_default_dtype(torch.float64)

fem = FEA.FEA_INP()
# fem.Read_INP(
#     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
# )

# fem.Read_INP(
#     'Z:\RESULT\T20240325195025_\Cache/TopOptRun.inp'
# )

fem.Read_INP(current_path + '/C3D10.inp')

fe = FEA.from_inp(fem)

surface_pressure = fem.part['final_model'].surfaces['surface_1_All']


surf = []
for i in range(len(surface_pressure)):
    elem_ind = np.array(list(fem.part['final_model'].sets_elems[surface_pressure[i][0]]))
    surf_element = fe.get_surface_triangles(elem_ind=elem_ind, surf_ind=surface_pressure[i][1])
    surf += surf_element

surf = torch.cat(surf, dim=0).cpu().numpy()
print(surf.shape)
print(fem.part['final_model'].surfaces_tri['surface_1_All'].shape)
