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

fem.Read_INP(current_path + '/C3D6.inp')

fe = FEA.from_inp(fem)



bc_dof = np.array(
    list(fem.part['Part-1'].sets_nodes['bottom'])) * 3
bc_dof = np.concatenate([bc_dof, bc_dof + 1, bc_dof + 2])
fe.add_constraint(
    FEA.constraints.Boundary_Condition(indexDOF=bc_dof,
                                    dispValue=torch.zeros(bc_dof.size)))

rp = fe.add_reference_point(FEA.ReferencePoint([0, 0, 50]))
indexNodes = np.where((abs(fe.nodes[:, 2] - 50)
                        < 0.1).cpu().numpy())[0]


fe.add_constraint(FEA.constraints.Couple(indexNodes=indexNodes, rp_name=rp))
fe.add_load(
    FEA.loads.Concentrate_Force(rp_name=rp, force=[0., -1., 0.]))
t1 = time.time()
fe.initialize()

fe.solve(tol_error=1e-8)

print(fe.GC[-6:])
print('ok')