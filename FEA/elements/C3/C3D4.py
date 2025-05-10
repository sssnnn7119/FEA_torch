import numpy as np
import torch
from .C3base import Element_3D

class C3D4(Element_3D):
    """
        Local coordinates:
            origin: 0-th nodal
            \ksi_0: 0-1 vector
            \ksi_1: 0-2 vector
            \ksi_2: 0-3 vector

        face nodal always point at the void
            face0: 021
            face1: 013
            face2: 123
            face3: 032

        shape_funtion:
            N_i = \ksi_i * \ksi_i, i<=3
    """

    def __init__(self, elems: torch.Tensor = None, elems_index: torch.Tensor = None):
        super().__init__(elems=elems, elems_index=elems_index)

        self.order = 1
        


    def initialize(self, fea):
        
        self.shape_function = [
            torch.tensor([[1.0, -1.0, -1.0, -1.0], [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            torch.tensor([[[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                          [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                          [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
        ]

        self.num_nodes_per_elem = 4
        self._num_gaussian = 1
        self.gaussian_weight = torch.tensor([1 / 6])

        p0 = torch.tensor([[0.25, 0.25, 0.25]])
        self._pre_load_gaussian(p0, nodes=fea.nodes)
        super().initialize(fea)
        
    
    def find_surface(self, surface_ind: int,
                           elems_ind: torch.Tensor):

        index_now = np.where(np.isin(self._elems_index, elems_ind))[0]

        if surface_ind == 0:
            return self._elems[index_now][:, [0, 2, 1]]
        elif surface_ind == 1:
            return self._elems[index_now][:, [0, 1, 3]]
        elif surface_ind == 2:
            return self._elems[index_now][:, [1, 2, 3]]
        elif surface_ind == 3:
            return self._elems[index_now][:, [0, 3, 2]]
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")
