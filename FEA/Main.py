import os
import sys
import time
import ctypes
import numpy as np
import torch

from . import loads
from . import elements
from . import constraints
from .reference_points import ReferencePoint

class FEA_Main():
    """
    Main class for Finite Element Analysis (FEA).
    This class handles the core functionality of finite element analysis, including
    model initialization, solving, and post-processing. It manages nodes, elements,
    materials, loads, and constraints to simulate structural behavior.
    Attributes:
        nodes (list): List containing node coordinates and reference points.
        elems (dict): Dictionary mapping element types to Element objects.
        RGC_list_indexStart (list): Start indices for redundant generalized coordinates.
        _RGC_nameMap (dict): Maps RGC indices to names.
        _RGC_size (dict): Maps RGC indices to size information.
        loads (dict): Dictionary of applied loads.
        constraints (dict): Dictionary of applied constraints.
        RGC (list): Redundant generalized coordinates.
        RGC_remain_index (list): Boolean masks for remaining indices.
        GC (torch.Tensor): Generalized coordinates.
        GC_list_indexStart (list): Start indices for generalized coordinates.
        RGC_remain_index_flatten (torch.Tensor): Flattened boolean mask for remaining indices.
        node_sets (dict): Dictionary of node sets imported from FEA_INP.
        element_sets (dict): Dictionary of element sets imported from FEA_INP.
        surface_sets (dict): Dictionary of surface sets imported from FEA_INP.
    Methods:
        initialize: Initialize the finite element model.
        solve: Solve the finite element analysis problem.
        assemble_force: Assemble the global force vector.
        add_reference_point: Add a reference point to the model.
        add_load: Add a load to the model.
        add_constraint: Add a constraint to the model.
        solve_linear_perturbation: Solve a linear perturbation problem.
        add_node_set: Add a node set to the model.
        add_element_set: Add an element set to the model.
        add_surface_set: Add a surface set to the model.
        get_node_set: Get a node set by name.
        get_element_set: Get an element set by name.
        get_surface_set: Get a surface set by name.
    """

    def __init__(self, nodes: torch.Tensor) -> None:
        """
        Initialize the FEA class.

        Args:
            nodes (torch.Tensor): The nodes of the finite element model.
        """
        self.nodes: torch.Tensor = nodes
        """
        record the nodes of the finite element model.\n
        """

        self.RGC_list_indexStart: list[int]

        # initialize the reference points
        self.reference_points: dict[str, ReferencePoint] = {}

        # initialize the elements
        self.elems: dict[str, elements.BaseElement] = {}
        """
        record the elements of the finite element model.\n
        """

        # initialize the loads
        self.loads: dict[str, loads.BaseLoad] = {}

        # initialize the constraints
        self.constraints: dict[str, constraints.BaseConstraint] = {}
        
        # initialize sets collections
        self.node_sets: dict[str, np.ndarray] = {}
        self.element_sets: dict[str, np.ndarray] = {}  
        self.surface_sets: dict[str, np.ndarray] = {}
        
        self._RGC_nameMap: dict[int, str]
        """
        record the name of the RGC\n
        {0: 'nodes_disp', 1: 'nodes_rot'}
        """

        self._RGC_size: dict[int, list[int]]
        """
        record the size of the RGC\n
        {0: nodes.shape, 1: nodes.shape}
        """
        
        self.RGC: list[torch.Tensor]
        """
        record the redundant generalized coordinates\n
        [0]: translation\n
        [1]: orientation\n
        [2]: allocated for other objects\n
        """

        self.RGC_remain_index: list[np.ndarray]
        """
        record the remaining index of the RGC\n
        """

        self.RGC_remain_index_flatten: torch.Tensor
        """
        record the remaining index of the RGC (flattened)\n
        """

        # initialize the GC (generalized coordinates)
        self.GC: torch.Tensor
        """
        record the generalized coordinates\n
        """

        self._GC_list_indexStart: list[int] = []
        """
        record the start index of the GC\n
        """

    def initialize(self, RGC0: torch.Tensor = None):
        """
        Initialize the finite element model.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.

        Returns:
            None
        """

        # region initialize the RGC

        # initialize the RGC (redundant generalized coordinate)
        self.RGC: list[torch.Tensor] = [
            torch.zeros_like(self.nodes),
            torch.zeros_like(self.nodes)
        ]

        self.RGC_remain_index: list[np.ndarray] = [
            np.zeros(self.nodes.shape, dtype=bool),
            np.zeros(self.nodes.shape, dtype=bool)
        ]

        self._RGC_nameMap: dict[int, str] = {0: 'nodes_disp', 1: 'nodes_rot'}
        self._RGC_size: dict[int, list[int]] = {0: self.nodes.shape, 1: self.nodes.shape}

        for rp in self.reference_points.keys():
            RGC_index = self._allocate_RGC(size=self.reference_points[rp]._RGC_requirements, name=rp)
            self.reference_points[rp].set_RGC_index(RGC_index)

        for e in self.elems.keys():
            RGC_index = self._allocate_RGC(size=self.elems[e]._RGC_requirements, name=e)
            self.elems[e].set_RGC_index(RGC_index)

        for f in self.loads.keys():
            RGC_index = self._allocate_RGC(size=self.loads[f]._RGC_requirements, name=f)
            self.loads[f].set_RGC_index(RGC_index)

        for c in self.constraints.keys():
            RGC_index = self._allocate_RGC(size=self.constraints[c]._RGC_requirements, name=c)
            self.constraints[c].set_RGC_index(RGC_index)

        self.RGC_list_indexStart = [0]
        for i in range(len(self.RGC)):
            self.RGC_list_indexStart.append(self.RGC_list_indexStart[i] + self.RGC[i].numel())
        
        if RGC0 is not None:
            for i in range(min(len(self.RGC), len(RGC0))):
                self.RGC[i] = RGC0[i].clone().detach()
        # endregion

        # region initialize the elements, loads, and constraints
        # initialize the elements
        for e in self.elems.values():
            e.initialize(self)

        # initialize the loads
        for l in self.loads.values():
            l.initialize(self)

        # initialize the constraints
        for c in self.constraints.values():
            c.initialize(self)

        # endregion

        # region modify the RGC_remain_index
        for e in self.elems.values():
            self.RGC_remain_index = e.set_required_DoFs(
                self.RGC_remain_index)
            
        for f in self.loads.values():
            self.RGC_remain_index = f.set_required_DoFs(
                self.RGC_remain_index)
        
        for c in self.constraints.values():
            self.RGC_remain_index = c.set_required_DoFs(
                self.RGC_remain_index)

        self.RGC_remain_index_flatten = np.concatenate([
            self.RGC_remain_index[i].reshape(-1)
            for i in range(len(self.RGC_remain_index))
        ]).tolist()
        self.RGC_remain_index_flatten = torch.tensor( self.RGC_remain_index_flatten, dtype=torch.bool)

        # GC core
        self.GC = self._RGC2GC(self.RGC)
        self._GC_list_indexStart = np.cumsum([
            self.RGC_remain_index[j].sum()
            for j in range(len(self.RGC_remain_index))
        ]).tolist()
        self._GC_list_indexStart.insert(0, 0)

        # endregion

    def solve(self,
              RGC0: torch.Tensor = None,
              tol_error: float = 1e-7):
        """
        Solves the finite element analysis problem.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.
            tol_error (float, optional): Tolerance error for convergence. Defaults to 1e-7.

        Returns:
            torch.Tensor: The solution vector.
        """
        # initialize the RGC
        self.initialize(RGC0=RGC0)

        # initialize the iteration
        iter_total = 0

        # initialize the load percentage
        load_percentage_now = 1.0
        load_percentage_finished = 0.0

        t0 = time.time()
        # start the iteration
        while True:
            # solve the iteration
            load_percentage = load_percentage_now + load_percentage_finished
            if load_percentage == 1.0:
                tol_now = tol_error
            else:
                tol_now = tol_error * 10
            if not self._solve_iteration(RGC=self.RGC, tol_error=tol_now, load_percentage=load_percentage):
                load_percentage_now *= 0.5
                if load_percentage_now < 1e-3:
                    print('load percentage is too small')
                    break
                continue

            t1 = time.time()

            # stop condition
            if self.GC.abs().max() < tol_error:
                break
            iter_total += 1

            # print the information
            print('total_iter:%d, total_time:%.2f\n' % (iter_total, t1 - t0))

            # update the RGC
            self.RGC = self._GC2RGC(self.GC)

            # update the load percentage
            load_percentage_finished += load_percentage_now
            load_percentage_now = 1.0 - load_percentage_finished

            # if load percentage is finished, break the loop
            if load_percentage_now == 0:
                break
        
        RGC_out = []
        for i in range(len(self.RGC)):
            RGC_out.append(self.RGC[i].clone().detach())
        
        return RGC_out

    # region solve iteration

    def assemble_force(self, force, GC0: torch.Tensor = None) -> torch.Tensor:
        if force.dim() == 1:
            force = force.unsqueeze(0)
        R = force.clone()
        if GC0 is None:
            GC0 = self.GC
        RGC = self._GC2RGC(GC0)
        for c in self.constraints.values():
            for i in range(R.shape[0]):
                R_new = c.modify_R(RGC, force[i].flatten())
                R[i] += R_new
                
        R = R[:, self.RGC_remain_index_flatten]
        
        return R
    
    def _assemble_Stiffness_Matrix(self,
                                   RGC: list[torch.Tensor],
                                   load_percentage: float = 1.0):
        """
        Assemble the stiffness matrix.

        Args:
            RGC (list[torch.Tensor]): The redundant generalized coordinates.
            load_percentage (float, optional): The load percentage. Defaults to 1.0.

        Returns:
            tuple: A tuple containing the right-hand side vector, the indices of the stiffness matrix, and the values of the stiffness matrix.
        """
        #region evaluate the structural K and R
        R0, K_indices, K_values = self._assemble_generalized_Matrix(RGC, load_percentage)
        # endregion
        R, K_indices, K_values = self._assemble_reduced_Matrix(RGC, R0, K_indices, K_values)

        return R, K_indices, K_values

    def _assemble_generalized_Matrix(self,
                                   RGC: list[torch.Tensor],
                                   load_percentage: float = 1.0):

        #region evaluate the structural K and R
        t0 = time.time()
        K_values = []
        K_indices = []
        R_values = []
        R_indices = []
        
        for e in self.elems.values():
            Ra_indice, Ra_values, Ka_indice, Ka_value = e.structural_Force(RGC=RGC)
            K_values.append(Ka_value)
            K_indices.append(Ka_indice)
            R_values.append(Ra_values)
            R_indices.append(Ra_indice)
        t1 = time.time()
        for f in self.loads.values():
            Rf_indice, Rf_values, Kf_indice, Kf_value = f.get_stiffness(RGC=RGC)
            K_values.append(-load_percentage * Kf_value)
            K_indices.append(Kf_indice)
            R_values.append(-load_percentage * Rf_values)
            R_indices.append(Rf_indice)
        t2 = time.time()
        # endregion

        K_indices = torch.cat(K_indices, dim=1)
        K_values = torch.cat(K_values, dim=0)
        R_indices = torch.cat(R_indices, dim=0)
        R_values = torch.cat(R_values, dim=0)

        R0 = torch.zeros(self.RGC_list_indexStart[-1])
        # Convert R_indices to int64 explicitly for scatter operation
        R0.scatter_add_(0, R_indices.to(torch.int64), R_values)
        return R0, K_indices, K_values

    def _assemble_reduced_Matrix(self, RGC: list[torch.Tensor], R0: torch.Tensor, K_indices: torch.Tensor, K_values: torch.Tensor):
        t0 = time.time()
        R = R0.clone()
        #region consider the constraints
        for c in self.constraints.values():
            R_new, Kc_indices, Kc_values = c.modify_R_K(
                RGC, R0, K_indices, K_values)
            K_indices = torch.cat([K_indices, Kc_indices], dim=1)
            K_values = torch.cat([K_values, Kc_values])
            R += R_new
        t4 = time.time()
        #endregion

        # get the global stiffness matrix and force vector
        index_remain = self.RGC_remain_index_flatten[K_indices[0].cpu()] & self.RGC_remain_index_flatten[K_indices[1].cpu()]
        K_values = K_values[index_remain]
        K_indices = K_indices[:, index_remain]
        t44 = time.time()
        
        K_indices[0] = K_indices[0].unique(return_inverse=True)[1]
        K_indices[1] = K_indices[1].unique(return_inverse=True)[1]

        t5 = time.time()

        R = R[self.RGC_remain_index_flatten]

        t6 = time.time()
        return R, K_indices, K_values

    def _total_Potential_Energy(self,
                                RGC: list[torch.Tensor],
                                load_percentage: float = 1.0):
        """
        Calculate the total potential energy of the finite element model.

        Args:
            RGC (list[torch.Tensor]): The redundant generalized coordinates.
            load_percentage (float, optional): The load percentage. Defaults to 1.0.

        Returns:
            float: The total potential energy.
        """

        # structural energy
        energy = 0
        for e in self.elems.values():
            energy = energy + e.potential_Energy(RGC=RGC)

        # force potential
        for f in self.loads.values():
            energy = energy - load_percentage * f.get_potential_energy(RGC=RGC)

        return energy

    def _line_search(self,
                     GC0,
                     dGC,
                     R,
                     energy0,
                     tol_error,
                     load_percentage=1.0):
        # line search
        alpha = 1
        beta = float('inf')
        c1 = 0.1
        c2 = 0.4
        dGC0 = dGC.clone()
        deltaE = (dGC * R).sum()

        if deltaE > 0:
            dGC = -dGC
            deltaE = -deltaE

        if torch.isnan(dGC).sum() > 0 or torch.isinf(dGC).sum() > 0:
            dGC = -R
            deltaE = (dGC * R).sum()

        # if abs(deltaE / energy0) < tol_error:
        #     return 1, GC0

        loopc2 = 0
        while True:
            GCnew = GC0 + alpha * dGC
            # GCnew.requires_grad_()
            energy_new = self._total_Potential_Energy(
                RGC=self._GC2RGC(GCnew), load_percentage=load_percentage)

            if torch.isnan(energy_new) or torch.isinf(
                    energy_new) or energy_new > energy0 + c1 * deltaE * alpha:
                alpha = 0.5 * alpha
                if alpha < 1e-10:
                    alpha = 0.0
                    break
            else:
                # Rnew = -torch.autograd.grad(energy_new, GCnew)[0]
                # if torch.dot(Rnew, dGC) > c2 * deltaE:
                #     beta = alpha
                #     alpha = 0.6 * (alpha + beta)
                # elif torch.dot(Rnew, dGC) < -c2 * deltaE:
                #     beta = alpha
                #     alpha = 0.4 * (alpha + beta)
                # else:
                    break
            loopc2 += 1
            if loopc2 > 20:
                c2 = 1000000000000000

        # if abs(alpha) < 1e-3:
        #     # gradient direction line search
        #     alpha = 1
        #     dGC = R
        #     while True:
        #         GCnew = GC0 + alpha * dGC
        #         energy_new = self._total_Potential_Energy(
        #             RGC=self._GC2RGC(GCnew), load_percentage=load_percentage)
        #         if energy_new < energy0:
        #             # pressure *= 1.2
        #             # pressure = min(pressure0, pressure)
        #             break
        #         alpha *= 0.8
        #         if abs(alpha) < tol_error:
        #             break

        # if abs(alpha) < 1e-3:
        #     alpha = 1
        #     GCnew = GC0 + alpha * dGC0
        return alpha, GCnew.detach(), energy_new.detach()

    def _solve_iteration(self, RGC: list[torch.Tensor], tol_error: float, load_percentage = 1.0):

        GC = self._RGC2GC(RGC)
            
        # iteration now
        iter_now = 0

        # initialize the time
        t00 = time.time()

        # initialize the energy
        energy = [
            self._total_Potential_Energy(RGC=RGC,
                                         load_percentage=load_percentage)
        ]

        # check the initial energy, if nan, reinitialize the RGC
        if torch.isnan(energy[-1]):
            for i in range(len(self.RGC)):
                RGC[i] = torch.randn_like(self.RGC[i]) * 1e-10

            GC = self._RGC2GC(RGC)
            energy[-1] = self._total_Potential_Energy(
                RGC=RGC, load_percentage=load_percentage)
        dGC = torch.zeros_like(GC)

        # record the number of low alpha
        low_alpha = 0
        alpha = 0

        # begin the iteration
        while True:

            # calculate the force vector and tangential stiffness matrix
            t1 = time.time()
            R, K_indices, K_values = self._assemble_Stiffness_Matrix(
                RGC=RGC, load_percentage=load_percentage)

            # stop condition
            # if R.abs().max() < tol_error:
            #     break
            iter_now += 1

            # evaluate the newton direction
            t2 = time.time()
            dGC = self._solve_linear_equation(K_indices=K_indices,
                                              K_values=K_values,
                                              R=-R,
                                              iter_now=iter_now,
                                              alpha0=alpha,
                                              tol_error=tol_error,
                                              dGC0=dGC).flatten()
            
            if dGC.abs().max() < tol_error:
                break
            
            # dGC = torch.linalg.solve(
            #     torch.sparse_coo_tensor(K_indices, K_values, [R.shape[0], R.shape[0]]).to_sparse_csr().to_dense(),
            #     R.flatten()).flatten()

            # line search
            t3 = time.time()
            alpha, GCnew, energynew = self._line_search(
                GC, dGC, R, energy[-1], tol_error, load_percentage)

            if abs(alpha) < 1e-6:
                print('max error:%e' % (R.abs().max()))
                break

            # if convergence has difficulty, reduce the load percentage
            if alpha < 0.1 and (R.abs().max()) > 1e0:
                low_alpha += 1
            else:
                low_alpha = 0
            if low_alpha > 300:
                return False

            # update the GC
            GC = GCnew

            # update the RGC
            RGC = self._GC2RGC(GC)

            # update the energy
            energynew = self._total_Potential_Energy(
                RGC=RGC, load_percentage=load_percentage)
            energy.append(energynew)

            t4 = time.time()

            # return the index to the first line
            if iter_now > 0:
                print('\033[1A', end='')
                print('\033[1A', end='')
                print('\033[K', end
                        ='')
            
            print(  "{:^8}".format("iter") + \
                    "{:^8}".format("alpha") + \
                    "{:^15}".format("total") + \
                    "{:^15}".format("energy") + \
                    "{:^15}".format("error") + \
                    "{:^15}".format("assemble") + \
                    "{:^15}".format("linearEQ") + \
                    "{:^15}".format("line search") + \
                    "{:^15}".format("step"))

            print(  "{:^8}".format(iter_now) + \
                    "{:^8.2f}".format(alpha) + \
                    "{:^15.2f}".format(t4 - t00) + \
                    "{:^15.4e}".format(energy[-1]) + \
                    "{:^15.4e}".format(R.abs().max()) + \
                    "{:^15.2f}".format(t2 - t1) + \
                    "{:^15.2f}".format(t3 - t2) + \
                    "{:^15.2f}".format(t4 - t3) + \
                    "{:^15.2f}".format(t4 - t1))

        self.GC = GC
        return True

    def _solve_iteration_opt(self, GC0: torch.Tensor, tol_error: float, load_percentage = 1.0):

        # iteration now
        iter_now = 0

        # initialize GC
        GC = GC0.clone()
        RGC = self._GC2RGC(GC)

        # initialize the time
        t00 = time.time()

        # initialize the energy
        energy = [
            self._total_Potential_Energy(RGC=RGC,
                                         load_percentage=load_percentage)
        ]

        # check the initial energy, if nan, reinitialize the RGC
        if torch.isnan(energy[-1]):
            for i in range(len(self.RGC)):
                RGC[i] = torch.zeros_like(self.RGC[i])
            # for reference point
            GC = self._RGC2GC(RGC)
            energy[-1] = self._total_Potential_Energy(
                RGC=RGC, load_percentage=load_percentage)
            
        dGC = torch.zeros_like(GC)
        
        def closure(x):
            R = -self._assemble_Stiffness_Matrix(RGC=self._GC2RGC(x + self.GC), load_percentage=load_percentage)[0]
            return R

        # record the number of low alpha
        low_alpha = 0
        alpha = 0

        # begin the iteration
        while True:
            # calculate the force vector and tangential stiffness matrix
            t1 = time.time()
            R, K_indices, K_values = self._assemble_Stiffness_Matrix(
                RGC=RGC, load_percentage=load_percentage)

            # stop condition
            if R.abs().max() < tol_error:
                break
            iter_now += 1

            # evaluate the newton direction
            t2 = time.time()
            
            # CG
            x = torch.zeros_like(self.GC)
            r = -closure(x)
            p = r
            rsold = (r**2).sum()

            for i in range(10000):
                Ap = torch.autograd.functional.jvp(closure, (x,), (p,))[1]
                alpha = rsold / (p * Ap).sum()
                x = x + alpha * p
                r = r - alpha * Ap
                rsnew = (r**2).sum()
                if rsnew < 1e-14:
                    break
                p = r + rsnew / rsold * p
                rsold = rsnew
                if i %100 == 0:
                    print('\riter: %d, residual: %e' % (i, rsnew), end='')

            # line search
            t3 = time.time()
            alpha, GCnew, energynew = self._line_search(
                GC, x, R, energy[-1], tol_error, load_percentage)

            if abs(alpha) < tol_error:
                print('max error:%e' % (R.abs().max()))
                break

            # if convergence has difficulty, reduce the load percentage
            if alpha < 0.1 and (R.abs().max()) > 1e0:
                low_alpha += 1
            else:
                low_alpha = 0
            if low_alpha > 300:
                return False

            # update the GC
            GC = GCnew

            # update the RGC
            RGC = self._GC2RGC(GC)

            # update the energy
            energynew = self._total_Potential_Energy(
                RGC=RGC, load_percentage=load_percentage)
            energy.append(energynew)

            t4 = time.time()

            print(  "{:^8}".format("iter") + \
                    "{:^8}".format("alpha") + \
                    "{:^15}".format("total") + \
                    "{:^15}".format("energy") + \
                    "{:^15}".format("error") + \
                    "{:^15}".format("assemble") + \
                    "{:^15}".format("linearEQ") + \
                    "{:^15}".format("line search") + \
                    "{:^15}".format("step"))



            print(  "{:^8}".format(iter_now) + \
                    "{:^8.2f}".format(alpha) + \
                    "{:^15.2f}".format(t4 - t00) + \
                    "{:^15.4e}".format(energy[-1]) + \
                    "{:^15.4e}".format(R.abs().max()) + \
                    "{:^15.2f}".format(t2 - t1) + \
                    "{:^15.2f}".format(t3 - t2) + \
                    "{:^15.2f}".format(t4 - t3) + \
                    "{:^15.2f}".format(t4 - t1))

        self.GC = GC
        return True

    __low_alpha_count = 0
    def _solve_linear_equation(self,
                               K_indices: torch.Tensor,
                               K_values: torch.Tensor,
                               R: torch.Tensor,
                               iter_now: int = 0,
                               alpha0: float = None,
                               dGC0: torch.Tensor = None,
                               tol_error=1e-8):
        if dGC0 is None:
            dGC0 = torch.zeros_like(R)

        if alpha0 is None:
            alpha0 = 1e-10
        
        # result = torch.sparse.spsolve(torch.sparse_coo_tensor(K_indices, K_values, [R.shape[0], R.shape[0]]).to_sparse_csr(), R)
            
        # precondition for the linear equation
        index = torch.where(K_indices[0] == K_indices[1])[0]
        diag = torch.zeros_like(R).scatter_add(0, K_indices[0,index], torch.sqrt(K_values[index]))
        K_values_preconditioned = K_values / diag[K_indices[0]]
        K_values_preconditioned = K_values_preconditioned / diag[K_indices[1]]        
        R_preconditioned = R / diag
        x0 = dGC0 * diag
        # x0 = R_preconditioned
        # x0 = torch.zeros_like(R)

        # record the number of low alpha
        if alpha0 < 1e-2:
            self.__low_alpha_count += 1
        
        # dx = _Linear_Solver.torch_solver(K_indices,
        #                                 K_values_preconditioned,
        #                                 R_preconditioned)
        # dx = _Linear_Solver.scipy_solver(K_indices,
        #                                 K_values_preconditioned,
        #                                 R_preconditioned)
        

        if self.__low_alpha_count > 3 or R_preconditioned.abs().max() < 1e-3 :
            dx = _Linear_Solver.conjugate_gradient(K_indices,
                                                       K_values_preconditioned,
                                                       R_preconditioned,
                                                       x0,
                                                       tol=1e-5,
                                                       max_iter=12000)
            # dx = _Linear_Solver.pypardiso_solver(K_indices,
            #                                        K_values_preconditioned,
            #                                        R_preconditioned)
            self.__low_alpha_count = 0
        else:
            if iter_now % 20 == 0:
                dx = _Linear_Solver.conjugate_gradient(K_indices,
                                                       K_values_preconditioned,
                                                       R_preconditioned,
                                                       x0,
                                                       tol=1e-3,
                                                       max_iter=6000)
            else:
                 dx = _Linear_Solver.conjugate_gradient(K_indices,
                                                       K_values_preconditioned,
                                                       R_preconditioned,
                                                       x0,
                                                       tol=1e-3,
                                                       max_iter=2000)
        result = dx.to(R.dtype) / diag
        return result

    # endregion
    
    # region create CAD model

    def add_reference_point(self, rp: ReferencePoint, name: str = None):
        """
        Adds a reference point to the FEA object.

        Parameters:
            node (torch.Tensor): The node to be added as a reference point.

        Returns:
            str: The name of the reference point.
        """

        if name is None:
            number = len(self.reference_points)
            while ( 'rp-%d' % number) in self.reference_points:
                number += 1
            name = 'rp-%d' % number

        self.reference_points[name] = rp

        return name
    
    def delete_reference_point(self, name: str):
        """
        Deletes a reference point from the FEA object.

        Parameters:
        - name (str): The name of the reference point to be deleted.

        Returns:
        - None
        """
        if name in self.reference_points:
            del self.reference_points[name]
        else:
            raise ValueError(f"Reference point '{name}' not found in the model.")

    def add_load(self, load: loads.BaseLoad, name: str = None):
        """
        Add a load to the FEA model.

        Parameters:
            load (Load.Force_Base): The load to be added.

        Returns:
            str: The name of the load.
        """
        if name is None:
            number = len(self.loads)
            while ( 'load-%d' % number) in self.loads:
                number += 1
            name = 'load-%d' % number
        self.loads[name] = load

        return name

    def delete_load(self, name: str):
        """
        Delete a load from the FEA model.

        Parameters:
            name (str): The name of the load to be deleted.

        Returns:
            None
        """
        if name in self.loads:
            del self.loads[name]
        else:
            raise ValueError(f"Load '{name}' not found in the model.")

    def add_constraint(self,
                       constraint: constraints.BaseConstraint,
                       name: str = None):
        """
        Add a constraint to the FEA model.

        Parameters:
            constraint (Constraints.Constraints_Base): The constraint to be added.

        Returns:
            str: The name of the constraint.
        """
        if name is None:
            number = len(self.constraints)
            while ( 'constraint-%d' % number) in self.constraints:
                number += 1
            name = 'constraint-%d' % number
        self.constraints[name] = constraint
        return name
    
    def delete_constraint(self, name: str):
        """
        Delete a constraint from the FEA model.

        Parameters:
            name (str): The name of the constraint to be deleted.

        Returns:
            None
        """
        if name in self.constraints:
            del self.constraints[name]
        else:
            raise ValueError(f"Constraint '{name}' not found in the model.")

    def add_element(self, element: elements.BaseElement, name: str = None):
        """
        Add an element to the FEA model.

        Parameters:
            element (elements.Element_Base): The element to be added.

        Returns:
            str: The name of the element.
        """
        if name is None:
            number = len(self.elems)
            while ( 'element-%d' % number) in self.elems:
                number += 1
            name = 'element-%d' % number
        self.elems[name] = element
        return name
    
    def delete_element(self, name: str):
        """
        Delete an element from the FEA model.

        Parameters:
            name (str): The name of the element to be deleted.

        Returns:
            None
        """
        if name in self.elems:
            del self.elems[name]
        else:
            raise ValueError(f"Element '{name}' not found in the model.")
    
    def refine_RGC(self):
        for e in self.elems.values():
            e.refine_RGC(self.RGC, self.nodes)
            
    # endregion

    # region interface for GC
    def _allocate_RGC(self, size: list[int], name: str = None):
        """
        Allocate memory for the RGC data structure.

        Args:
        - size: A list of integers representing the size of the RGC tensor.
        - name: (optional) A string representing the name of the RGC tensor.

        Returns:
        None
        """
        
        index_now = max(list(self._RGC_size.keys())) + 1

        if name is None:
            name = 'RGC-%d' % index_now

        self._RGC_nameMap[index_now] = name
        self._RGC_size[index_now] = size

        self.RGC.append(torch.randn(size) * 1e-10)
        self.RGC_remain_index.append(np.zeros(size, dtype=bool))

        return index_now

    def _GC2RGC(self, GC: torch.Tensor):
        """
        Converts the global control vector (GC) to the reduced global control vector (RGC).

        Args:
            GC (torch.Tensor): The global control vector.

        Returns:
            list: The reduced global control vector (RGC).
        """
        RGC = []
        for i in range(len(self.RGC_remain_index)):
            RGC.append(torch.zeros(self._RGC_size[i]))
            RGC[-1][self.RGC_remain_index[i]] = GC[
                self._GC_list_indexStart[i]:self._GC_list_indexStart[i + 1]]

        for c in self.constraints.values():
            RGC = c.modify_RGC(RGC)

        return RGC

    def _GC2RGC_linear(self, GC: torch.Tensor):
        """
        Converts the global control vector (GC) to the reduced global control vector (RGC).

        Args:
            GC (torch.Tensor): The global control vector.

        Returns:
            list: The reduced global control vector (RGC).
        """
        RGC = []
        for i in range(len(self.RGC_remain_index)):
            RGC.append(torch.zeros(self._RGC_size[i]))
            RGC[-1][self.RGC_remain_index[i]] = GC[
                self._GC_list_indexStart[i]:self._GC_list_indexStart[i + 1]]

        for c in self.constraints.values():
            RGC = c.modify_RGC_linear(RGC)

        return RGC

    def _RGC2GC(self, RGC: list[torch.Tensor]):
        GC = torch.cat([
            RGC[i][self.RGC_remain_index[i]].flatten() for i in range(len(RGC))
        ],
                       dim=0)
        return GC

    # endregion

    # region for linear perturbation


    def solve_linear_perturbation(
        self, R0: torch.Tensor = None, R : torch.Tensor = None,
        GC0: torch.Tensor = torch.zeros([0])) -> torch.Tensor:
        """
        Solve the linear perturbation problem.

        Args:
            R0 (torch.Tensor): The right-hand side vector.
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.

        Returns:
            torch.Tensor: The solution vector.
        """
        # initialize the RGC
        if GC0.numel() != 0:
            RGC = self._GC2RGC(GC0)
        else:
            RGC = self.RGC

        if R is None:
            if R0.dim() == 1:
                R0 = R0.unsqueeze(0)
            R = R0.clone()

            for c in self.constraints.values():
                for i in range(R.shape[0]):
                    R_new = c.modify_R(RGC, R0[i].flatten())
                    R[i] += R_new

            R = R[:, self.RGC_remain_index_flatten]

        if R.dim() == 1:
            R = R.unsqueeze(0)
        
        K_indices, K_values = self._assemble_Stiffness_Matrix(RGC=RGC)[1:]
        index = torch.where(K_indices[0] == K_indices[1])[0]
        diag = torch.zeros(R.shape[1]).scatter_add(0, K_indices[0,index], torch.sqrt(K_values[index]))
        K_values_preconditioned = K_values / diag[K_indices[0]]
        K_values_preconditioned = K_values_preconditioned / diag[K_indices[1]]        
        

        dGC = []
        for i in range(R.shape[0]):
            R_preconditioned = R[i] / diag
            dGC.append(
                _Linear_Solver.conjugate_gradient(K_indices, K_values_preconditioned,
                                                    R_preconditioned, tol=1e-13, max_iter=30000) / diag)
        dGC = torch.stack(dGC, dim=0)
        
        return dGC
    
    # endregion
    
    # region Sets Management
    def add_node_set(self, name: str, indices: np.ndarray):
        """
        Add a node set to the FEA model.
        
        Args:
            name (str): Name of the node set.
            indices (np.ndarray): Array of node indices.
            
        Returns:
            str: Name of the added node set.
        """
        self.node_sets[name] = np.array(indices, dtype=int)
        return name
    
    def add_element_set(self, name: str, indices: np.ndarray):
        """
        Add an element set to the FEA model.
        
        Args:
            name (str): Name of the element set.
            indices (np.ndarray): Array of element indices.
            
        Returns:
            str: Name of the added element set.
        """
        self.element_sets[name] = np.array(indices, dtype=int)
        return name
    
    def add_surface_set(self, name: str, elements: np.ndarray):
        """
        Add a surface set to the FEA model.
        
        Args:
            name (str): Name of the surface set.
            elements (np.ndarray): Surface elements information.
            
        Returns:
            str: Name of the added surface set.
        """
        self.surface_sets[name] = elements
        return name
    
    def get_node_set(self, name: str) -> np.ndarray:
        """
        Get a node set by name.
        
        Args:
            name (str): Name of the node set.
            
        Returns:
            np.ndarray: Array of node indices.
            
        Raises:
            KeyError: If the node set doesn't exist.
        """
        if name in self.node_sets:
            return self.node_sets[name]
        raise KeyError(f"Node set '{name}' not found in the model.")
    
    def get_element_set(self, name: str) -> np.ndarray:
        """
        Get an element set by name.
        
        Args:
            name (str): Name of the element set.
            
        Returns:
            np.ndarray: Array of element indices.
            
        Raises:
            KeyError: If the element set doesn't exist.
        """
        if name in self.element_sets:
            return self.element_sets[name]
        raise KeyError(f"Element set '{name}' not found in the model.")
    
    def get_surface_set(self, name: str) -> np.ndarray:
        """
        Get a surface set by name.
        
        Args:
            name (str): Name of the surface set.
            
        Returns:
            np.ndarray: Surface elements information.
            
        Raises:
            KeyError: If the surface set doesn't exist.
        """
        if name in self.surface_sets:
            return self.surface_sets[name]
        
        raise KeyError(f"Surface set '{name}' not found in the model.")
    
    def get_surface_triangles(self, name: str) -> np.ndarray:
        surface = []
        for surf_index in self.surface_sets[name]:
            elem_ind = surf_index[0]
            surf_ind = surf_index[1]
            for e in self.elems.values():
                s_now = e.find_surface(surf_ind, elem_ind)
                if s_now is not None:
                    surface.append(s_now)
        if len(surface) == 0:
            raise ValueError(f"Surface {surf_ind} not found in the model.")
        else:
            return surface
    
    def delete_node_set(self, name: str):
        """
        Delete a node set from the FEA model.
        
        Args:
            name (str): Name of the node set to delete.
            
        Raises:
            KeyError: If the node set doesn't exist.
        """
        if name in self.node_sets:
            del self.node_sets[name]
        else:
            raise KeyError(f"Node set '{name}' not found in the model.")
    
    def delete_element_set(self, name: str):
        """
        Delete an element set from the FEA model.
        
        Args:
            name (str): Name of the element set to delete.
            
        Raises:
            KeyError: If the element set doesn't exist.
        """
        if name in self.element_sets:
            del self.element_sets[name]
        else:
            raise KeyError(f"Element set '{name}' not found in the model.")
    
    def delete_surface_set(self, name: str):
        """
        Delete a surface set from the FEA model.
        
        Args:
            name (str): Name of the surface set to delete.
            
        Raises:
            KeyError: If the surface set doesn't exist.
        """
        if name in self.surface_sets:
            del self.surface_sets[name]
        else:
            raise KeyError(f"Surface set '{name}' not found in the model.")
            
    # endregion


class _Linear_Solver():

    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def conjugate_gradient(A_indices: torch.Tensor,
                            A_values: torch.Tensor,
                            b: torch.Tensor,
                            x0: torch.Tensor=torch.zeros([0]),
                            tol: float=1e-3,
                            max_iter: int=1500):
        # A_values = A_values.to(torch.float64)
        # b = b.to(torch.float64)
        # x0 = x0.to(torch.float64)
        if x0.numel() == 0:
            x0 = torch.zeros_like(b)
            
        A = torch.sparse_coo_tensor(A_indices, A_values, [b.shape[0], b.shape[0]]).to_sparse_csr()

        # reference error for convergence
        r_r0 = torch.dot(b, b)

        # 定义初始解x和残差r
        x = x0.clone()

        r = b - A @ x
        p = r
        rsold = torch.dot(r, r)

        for i in range(max_iter):
            Ap = A @ p
            alpha = rsold / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)
            if rsnew / r_r0 < tol:
                break
            p = r + rsnew / rsold * p
            rsold = rsnew

        return x
    
    @staticmethod
    def pypardiso_solver(A_indices: torch.Tensor,
                            A_values: torch.Tensor,
                            b: torch.Tensor):
        import pypardiso
        import scipy.sparse as sp

        A_sp = sp.coo_matrix((A_values.cpu().numpy(), (A_indices[0].cpu().numpy(), A_indices[1].cpu().numpy()))).tocsr()

        b_np = b.cpu().numpy()
        x = pypardiso.spsolve(A_sp, b_np)

        return torch.from_numpy(x).to(b.dtype).to(b.device)


def export_surface_stl(nodes, extern_surf, U):
    from mayavi import mlab
    import vtk
    from mayavi import mlab
    coo = extern_surf.tolist()

    # Get the deformed surface coordinates
    deformed_surface = nodes + U

    r = deformed_surface.transpose(0, 1).tolist()

    Unorm = ((U**2).sum(axis=1)**0.5).tolist()

    surface = mlab.pipeline.triangular_mesh_source(r[0], r[1], r[2], coo)
    surface_vtk = surface.outputs[0]._vtk_obj
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName('test.stl')
    stlWriter.SetInputConnection(surface_vtk.GetOutputPort())
    stlWriter.Write()
    mlab.close()

    # Plot the deformed surface
    mlab.triangular_mesh(r[0], r[1], r[2], coo, scalars=Unorm)

    mlab.show()


# if __name__ == '__main__':

#     torch.set_default_device(torch.device('cuda'))
#     torch.set_default_dtype(torch.float64)
#     from FEA_INP import FEA_INP

#     fem = FEA_INP()
#     fem.Read_INP(
#         'D:\MineData/Learning/Publications/TRO20230207Morph/20230411Revise/Result2/Bend0/FEA/c3d10.inp'
#     )

#     # fem.Read_INP(
#     #     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/TRO20230207Morph/20230411Revise/Result2/Twist0/FEA/c3d4.inp'
#     # )

#     # pressure_elems = fem.Find_Surface(['lateral1', 'bottom1', 'head1'])[1]
#     pressure_elems = fem.part['final_model'].surfaces['inner1']
#     FEA = FEA_Main(fem.part['final_model'].nodes[:, 1:],
#                    fem.part['final_model'].elems,
#                    fem.part['final_model'].elems_material)
#     FEA.add_load(Loads.Pressure(surface_element=pressure_elems, pressure=0.06))

#     # FEA.loads.append(Load.Moment(indexStart=FEA.RGC_list_indexStart, nodes=FEA.nodes, rp_index=2, moment=torch.tensor([0., 20., 0.])))

#     bc_dof = np.array(list(fem.part['final_model'].sets_nodes['bottom0'])) * 3
#     bc_dof = np.concatenate([bc_dof, bc_dof + 1, bc_dof + 2])
#     FEA.add_constraint(
#         Constraints.Boundary_Condition(indexDOF=bc_dof,
#                                        dispValue=torch.zeros(bc_dof.size)))

#     FEA.add_reference_point(node=torch.tensor([0, 0, 80]))
#     FEA.add_constraint(
#         Constraints.Couple(indexNodes=np.array(
#             list(fem.part['final_model'].sets_nodes['head0'])),
#                            rp_index=2))

#     # FEA.add_load(Load.Spring_Ground(dof=FEA.RGC_list_indexStart[-2] + 4, stiffness=100))

#     t1 = time.time()

#     FEA.solve()

#     t2 = time.time()

#     print('total time:%f' % (t2 - t1))

#     extern_surf = fem.Find_Surface(['lateral0'])[1]

#     export_surface_stl(FEA.nodes[0], extern_surf, FEA.RGC[0])

