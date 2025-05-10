
import numpy as np
import torch
from .base import BaseLoad

class Concentrate_Force(BaseLoad):

    def __init__(self, rp_name: str, force: list[float]) -> None:
        super().__init__()
        self.rp_name = rp_name
        self.rp_index: int = None
        self.force = torch.tensor(force)

    def initialize(self, fea):
        super().initialize(fea)
        self.rp_index = fea.reference_points[self.rp_name]._RGC_index
        self._indices_force = torch.arange(fea.RGC_list_indexStart[self.rp_index], fea.RGC_list_indexStart[self.rp_index]+3)

    def get_stiffness(self,
                RGC: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        return self._indices_force, self.force, torch.zeros([2, 0], dtype=torch.int), torch.zeros([0])

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        if type(self.force) == list:
            self.force = torch.tensor(self.force)
        return (self.force * RGC[self.rp_index][:3]).sum()

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[self.rp_index][:3] = True
        return RGC_remain_index
