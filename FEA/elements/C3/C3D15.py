import numpy as np
import torch
from .C3base import Element_3D

class C3D15(Element_3D):
    """
    # Local coordinates:
        origin: bottom triangle center
        g, h: coordinates in triangle base
        r: coordinate along prism height

    # Node numbering:
        - Bottom face (r=-1): 0, 1, 2 (vertices), 6, 7, 8 (mid-edge)
        - Top face (r=1): 3, 4, 5 (vertices), 9, 10, 11 (mid-edge)
        - Middle nodes (r=0): 12, 13, 14 (on vertical edges)

    # Face description:
        face0: 0(8)2(7)1(6) (Triangle)
        face1: 3(9)4(10)5(11) (Triangle)
        face2: 0(6)1(13)4(9)3(12) (Rectangle)
        face3: 1(7)2(14)5(10)4(13) (Rectangle)
        face4: 2(8)0(12)3(11)5(14) (Rectangle)

    # Shape functions:
        Quadratic interpolation in all directions
        Combines triangular base shape functions with prismatic extrusion
    """

    def __init__(self, elems: torch.Tensor = None, elems_index: torch.Tensor = None):
        super().__init__(elems=elems, elems_index=elems_index)
        self.order = 2
        # Shape function coefficients and derivatives
        # Format: [shape_function, derivatives]
        # These matrices represent the shape functions and their derivatives
        # for the 15-node prismatic element
    
    def initialize(self, fea):
        
        # Shape function matrix (coefficients for each node's shape function)
        self.shape_function = [
            # Shape functions for all 15 nodes
            torch.tensor([[0,-1.0,-1.0,-0.5,2.0,1.5,1.5,1.0,1.0,0.5,0,0,-1.0,-0.5,-0.5,-1.0,-2.0,0,0,0],
                [0,-1.0,0,0,0,0,0.5,1.0,0,0,0,0,0,0,0.5,-1.0,0,0,0,0],
                [0,0,-1.0,0,0,0.5,0,0,1.0,0,0,0,-1.0,0.5,0,0,0,0,0,0],
                [0,-1.0,-1.0,0.5,2.0,-1.5,-1.5,1.0,1.0,0.5,0,0,1.0,-0.5,-0.5,1.0,2.0,0,0,0],
                [0,-1.0,0,0,0,0,-0.5,1.0,0,0,0,0,0,0,0.5,1.0,0,0,0,0],
                [0,0,-1.0,0,0,-0.5,0,0,1.0,0,0,0,1.0,0.5,0,0,0,0,0,0],
                [0,2.0,0,0,-2.0,0,-2.0,-2.0,0,0,0,0,0,0,0,2.0,2.0,0,0,0],
                [0,0,0,0,2.0,0,0,0,0,0,0,0,0,0,0,0,-2.0,0,0,0],
                [0,0,2.0,0,-2.0,-2.0,0,0,-2.0,0,0,0,2.0,0,0,0,2.0,0,0,0],
                [0,2.0,0,0,-2.0,0,2.0,-2.0,0,0,0,0,0,0,0,-2.0,-2.0,0,0,0],
                [0,0,0,0,2.0,0,0,0,0,0,0,0,0,0,0,0,2.0,0,0,0],
                [0,0,2.0,0,-2.0,2.0,0,0,-2.0,0,0,0,-2.0,0,0,0,-2.0,0,0,0],
                [1.0,-1.0,-1.0,0,0,0,0,0,0,-1.0,0,0,0,1.0,1.0,0,0,0,0,0],
                [0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,-1.0,0,0,0,0,0],
                [0,0,1.0,0,0,0,0,0,0,0,0,0,0,-1.0,0,0,0,0,0,0]]),]
        
        self.shape_function.append(torch.stack([
            self._shape_function_derivative(self.shape_function[0], 0),
            self._shape_function_derivative(self.shape_function[0], 1),
            self._shape_function_derivative(self.shape_function[0], 2),
        ], dim=0))

        # Gauss weights

        gaussian_weight_triangle = torch.tensor([1/6, 1/6, 1/6])
        gaussian_points_triangle = torch.tensor([
            [1/6, 1/6],
            [2/3, 1/6],
            [1/6, 2/3]
        ])

        gaussian_weight_height = torch.tensor([5/9, 8/9, 5/9])
        gaussian_points_height = torch.tensor([-np.sqrt(3/5), 0, np.sqrt(3/5)])

        # Combine weights and points for 3D integration
        self.gaussian_weight = torch.einsum('i,j->ij', gaussian_weight_triangle, gaussian_weight_height).flatten()
        p0 = torch.cat([gaussian_points_triangle, torch.zeros([gaussian_points_triangle.shape[0], 1])], dim=1)
        p0 = p0.reshape([-1, 1, 3]).repeat([1, gaussian_points_height.shape[0], 1])
        p0[:, :, 2] = gaussian_points_height.reshape([1, -1])

        # Gauss integration points setup
        self.num_nodes_per_elem = 15
        self._num_gaussian = 9
        
        # Load the Gaussian points for integration
        self._pre_load_gaussian(p0.reshape([-1, 3]), nodes=self._nodes)
        super().initialize(fea)
        
    def find_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        index_now = np.where(np.isin(self._elems_index, elems_ind))[0]
        if surface_ind == 0:
            return Triangle_2nd(self._elems[index_now][:, [0, 2, 1, 8, 7, 6]], nodes)
        elif surface_ind == 1:
            return Triangle_2nd(self._elems[index_now][:, [3, 4, 5, 9, 10, 11]], nodes)
        elif surface_ind == 2:
            return Rectangle_2nd(self._elems[index_now][:, [0, 1, 4, 3, 6, 13, 9, 12]], nodes)
        elif surface_ind == 3:
            return Rectangle_2nd(self._elems[index_now][:, [1, 2, 5, 4, 7, 14, 10, 13]], nodes)
        elif surface_ind == 4:
            return Rectangle_2nd(self._elems[index_now][:, [2, 0, 3, 5, 8, 12, 11, 14]], nodes)
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")
        