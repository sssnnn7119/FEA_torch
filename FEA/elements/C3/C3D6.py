import numpy as np
import torch
from .C3base import Element_3D

class C3D6(Element_3D):
    """
    # Local coordinates:
        origin: 0-th nodal
        \ksi_0: 0-1 vector
        \ksi_1: 0-2 vector
        \ksi_2: 0-3 vector

    # face nodal always point at the void
        face0: 021 (Triangle)
        face1: 345 (Triangle)
        face2: 0143 (Rectangle)
        face3: 1254 (Rectangle)
        face4: 2035 (Rectangle)
    
    # shape_funtion:
        N_0 = 0.5 * (1 - \ksi_0 - \ksi_1) * (1 - \ksi_2) \n
        N_1 = 0.5 * \ksi_0 * (1 - \ksi_2) \n
        N_2 = 0.5 * \ksi_1 * (1 - \ksi_2) \n
        N_3 = 0.5 * (1 - \ksi_0 - \ksi_1) * (1 + \ksi_2) \n
        N_4 = 0.5 * \ksi_0 * (1 + \ksi_2) \n
        N_5 = 0.5 * \ksi_1 * (1 + \ksi_2) \n
    """
    
    def __init__(self, elems: torch.Tensor = None, elems_index: torch.Tensor = None):
        super().__init__(elems=elems, elems_index=elems_index)
        self.order = 1
    
    def initialize(self, fea):
        
        # Shape function coefficients in format aligned with your other elements
        self.shape_function = [
            torch.tensor([
                [0.5, -0.5, -0.5, -0.5, 0.0, 0.5, 0.5],
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -0.5],
                [0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0],
                [0.5, -0.5, -0.5, 0.5, 0.0, -0.5, -0.5],
                [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0]]),
            torch.tensor([
                [
                    [-0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ],
                [
                    [-0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0],
                    [-0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
                ],
                [
                    [-0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
                    [0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
            ]
        ])]
        self.num_nodes_per_elem = 6
        self._num_gaussian = 2
        self.gaussian_weight = torch.tensor([1 / 2, 1 / 2, ])

        # get the interpolation coordinates of the guass_points
        p0 = torch.tensor([[1/3, 1/3, 1 / np.sqrt(3)],
                           [1/3, 1/3, -1 / np.sqrt(3)]])
        
        self._pre_load_gaussian(p0, nodes=fea.nodes)
        super().initialize(fea)
        
    def find_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        index_now = np.where(np.isin(self._elems_index, elems_ind))[0]
        
        if surface_ind == 0:
            return self._elems[index_now][:, [0, 2, 1]]
        elif surface_ind == 1:
            return self._elems[index_now][:, [3, 4, 5]]
        elif surface_ind == 2:
            return torch.cat([self._elems[index_now][:, [0, 1, 4]], 
                              self._elems[index_now][:, [0, 4, 3]]], dim=1)
        elif surface_ind == 3:
            return torch.cat([self._elems[index_now][:, [1, 2, 5]],
                              self._elems[index_now][:, [1, 5, 4]]], dim=1)
        elif surface_ind == 4:
            return torch.cat([self._elems[index_now][:, [2, 0, 3]],
                              self._elems[index_now][:, [2, 3, 5]]], dim=1)
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")

