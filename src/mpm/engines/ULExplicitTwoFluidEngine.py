from src.mpm.boundaries.BoundaryCore import *
from src.mpm.engines.ULExplicitEngine import ULExplicitEngine
from src.mpm.engines.EngineKernel import *
from src.mpm.engines.TwoFluidKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.utils.linalg import no_operation


class ULExplicitTwoFluidEngine(ULExplicitEngine):
    """Explicit updated-Lagrangian MPM solver of the Eulerian-Eulerian water-sediment model.

    The balance equations are those of Shi Huabin, Eqs. (4.64)-(4.68).  Both phases share
    one set of material points which travel with the water velocity; the sediment velocity
    is an extra particle field integrated along the water trajectory.
    """

    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.reset_grid_messages = self.reset_grid_message

    def manage_function(self, sims: Simulation):
        super().manage_function(sims)
        self.compute_nodal_kinematic = self.compute_nodal_kinematics
        self.compute_forces = self.compute_force
        self.compute_particle_kinematic = no_operation
        self.compute_velocity_gradient = no_operation
        self.compute_stress_strains = no_operation
        self.pressure_smoothing_ = no_operation
        self.pre_contact_calculate = no_operation
        self.compute_contact_force_ = no_operation

    def reset_grid_message(self, scene: myScene):
        kernel_twofluid_grid_reset(scene.node)

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_twofluid_mass_momentum_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle,
                                          scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_grid_velcity(self, sims: Simulation, scene: myScene):
        kernel_twofluid_grid_velocity(scene.mass_cut_off, scene.node)

    def compute_force(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_twofluid_force_p2g(scene.element.grid_nodes, start_index, end_index, sims.gravity, scene.node, scene.particle,
                                      scene.material.materialID, scene.material.matProps[materialID + 1],
                                      scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)

    def compute_grid_kinematic(self, sims: Simulation, scene: myScene):
        kernel_twofluid_grid_kinematic(scene.mass_cut_off, sims.background_damping, scene.node, sims.dt)

    def compute_particle_kinematics(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_twofluid_g2p(scene.element.grid_nodes, sims.alphaPIC, sims.dt, start_index, end_index, scene.node, scene.particle,
                                scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars,
                                scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)

    def usl_updating(self, sims: Simulation, scene: myScene, neighbor=None):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematics(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.compute_force(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_particle_kinematics(sims, scene)

    def usf_updating(self, sims: Simulation, scene: myScene, neighbor=None):
        self.usl_updating(sims, scene, neighbor)

    def musl_updating(self, sims: Simulation, scene: myScene, neighbor=None):
        self.usl_updating(sims, scene, neighbor)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_twofluid_hydrostatic_state(start_index, end_index, scene.particle, scene.material.materialID,
                                              scene.material.matProps[materialID + 1], scene.material.stateVars)
        self.limit = sims.verlet_distance * sims.verlet_distance
