import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.MaterialModel import Fluid
from src.utils.constants import ZEROMAT2x2
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class TwoFluidSedimentModel(Fluid):
    """Eulerian-Eulerian (two-fluid) water-sediment model.

    The model follows Shi Huabin, *Mathematical models of water-sediment two phase
    flow and their applications*, Chapter 4 (Eqs. 4.31-4.78).  The original work
    discretises the equations with SPH; here the very same continuum model is
    discretised with the material point method, therefore only the closure
    relations (viscosities, drag and the water-sediment equation of state) live
    in this class while the balance equations are solved by
    :class:`src.mpm.engines.ULExplicitTwoFluidEngine.ULExplicitTwoFluidEngine`.
    """

    def __init__(self, material_type="TwoFluidSediment", configuration="UL", solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self.solid_density = 2650.
        self.particle_diameter = 8e-4
        self.water_viscosity = 1e-6
        self.smagorinsky = 0.1
        self.schmidt = 1.
        self.max_concentration = 0.65
        self.sound_speed = 30.
        self.tait_exponent = 7.
        self.damping_exponent = 5
        self.filter_scale = 0.005
        self.concentration = 0.
        self.background_concentration = 0.

    def model_initialize(self, material):
        density = DictIO.GetAlternative(material, 'Density', 1000.)
        solid_density = DictIO.GetAlternative(material, 'SolidDensity', 2650.)
        particle_diameter = DictIO.GetAlternative(material, 'ParticleDiameter', 8e-4)
        water_viscosity = DictIO.GetAlternative(material, 'WaterKinematicViscosity', 1e-6)
        smagorinsky = DictIO.GetAlternative(material, 'SmagorinskyCoefficient', 0.1)
        schmidt = DictIO.GetAlternative(material, 'SchmidtNumber', 1.)
        max_concentration = DictIO.GetAlternative(material, 'MaxConcentration', 0.65)
        sound_speed = DictIO.GetAlternative(material, 'SoundSpeed', 30.)
        tait_exponent = DictIO.GetAlternative(material, 'TaitExponent', 7.)
        damping_exponent = DictIO.GetAlternative(material, 'DampingExponent', 5)
        filter_scale = DictIO.GetAlternative(material, 'FilterScale', 0.005)
        element_length = DictIO.GetAlternative(material, 'ElementLength', 0.)
        concentration = DictIO.GetAlternative(material, 'Concentration', 0.)
        background = DictIO.GetAlternative(material, 'BackgroundConcentration', 0.)
        cl = DictIO.GetAlternative(material, 'cL', 0.5)
        cq = DictIO.GetAlternative(material, 'cQ', 0.)
        self.add_material(density, solid_density, particle_diameter, water_viscosity, smagorinsky, schmidt,
                          max_concentration, sound_speed, tait_exponent, damping_exponent, filter_scale,
                          element_length, concentration, background, cl, cq)

    def add_material(self, density, solid_density, particle_diameter, water_viscosity, smagorinsky, schmidt,
                     max_concentration, sound_speed, tait_exponent, damping_exponent, filter_scale,
                     element_length, concentration, background, cl, cq):
        if concentration < 0. or concentration >= max_concentration:
            raise RuntimeError(f"Keyword:: /Concentration/ must lie in [0, {max_concentration})")
        self.density = density
        self.solid_density = solid_density
        self.particle_diameter = particle_diameter
        self.water_viscosity = water_viscosity
        self.viscosity = water_viscosity * density
        self.smagorinsky = smagorinsky
        self.schmidt = schmidt
        self.max_concentration = max_concentration
        self.sound_speed = sound_speed
        self.tait_exponent = tait_exponent
        self.damping_exponent = damping_exponent
        self.filter_scale = filter_scale
        self.element_length = element_length
        self.concentration = max(concentration, background)
        self.background_concentration = background
        self.cl = cl
        self.cq = cq
        self.gamma = tait_exponent
        self.modulus = density * sound_speed * sound_speed
        # Eq. (4.78): the mixture celerity is largest for the densest admissible mixture
        self.max_sound_speed = self.mixture_sound_speed(max_concentration)

    def mixture_sound_speed(self, concentration):
        rho_f0 = self.density
        alpha_f_rho_f = (1. - concentration) * rho_f0
        summation = alpha_f_rho_f + concentration * rho_f0
        mixture = alpha_f_rho_f + concentration * self.solid_density
        return self.sound_speed * summation / np.sqrt(alpha_f_rho_f * mixture) * \
            (summation / rho_f0) ** (0.5 * (self.tait_exponent - 1.))

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model = Two-Fluid Water-Sediment Model')
        print("Model ID: ", materialID)
        print("Water reference density = ", self.density)
        print("Sediment density = ", self.solid_density)
        print("Sediment diameter = ", self.particle_diameter)
        print("Water kinematic viscosity = ", self.water_viscosity)
        print("Smagorinsky coefficient = ", self.smagorinsky)
        print("Sediment Schmidt number = ", self.schmidt)
        print("Maximum sediment concentration = ", self.max_concentration)
        print("Initial sediment concentration = ", self.concentration)
        print("Reference sound speed = ", self.sound_speed)
        print("Tait exponent = ", self.tait_exponent)
        print("Spatial filter scale = ", self.filter_scale, '\n')

    def define_state_vars(self):
        return {'pressure': float, 'concentration': float}

    @ti.func
    def _initialize_vars_(self, np, particle, stateVars):
        stateVars[np].pressure = particle[np].pressure
        stateVars[np].concentration = particle[np].alpha_s

    # ------------------------------------------------------------------ #
    #                         closure relations                          #
    # ------------------------------------------------------------------ #
    @ti.func
    def _pressure(self, alpha_f_rho_f, alpha_s):
        """Water-sediment mixture equation of state, Eq. (4.77)."""
        rho_f0 = self.density
        summation = alpha_f_rho_f + alpha_s * rho_f0
        ratio = summation / rho_f0
        return rho_f0 * self.sound_speed * self.sound_speed / self.tait_exponent * \
            (summation / alpha_f_rho_f) * (ti.pow(ratio, self.tait_exponent) - 1.)

    @ti.func
    def _compression_ratio(self, pressure, alpha_s0):
        """V / V0 obtained by inverting the Macdonald-Tait law, Eqs. (4.70), (4.76)."""
        bulk_modulus = self.density * self.sound_speed * self.sound_speed / ti.max(1. - alpha_s0, 1e-6)
        return ti.pow(1. + self.tait_exponent * ti.max(pressure, 0.) / bulk_modulus, -1. / self.tait_exponent)

    @ti.func
    def _artificial_pressure(self, density, volumetric_strain_rate):
        """Von Neumann - Richtmyer artificial viscosity, needed by any weakly
        compressible explicit scheme to damp the acoustic modes."""
        q = 0.
        if volumetric_strain_rate < 0.:
            q = -density * self.cl * self.element_length * volumetric_strain_rate + \
                density * self.cq * self.element_length * self.element_length * volumetric_strain_rate * volumetric_strain_rate
        return q

    @ti.func
    def _sediment_viscosity(self, alpha_s):
        """Ahilan & Sleath (1987) sediment viscosity, Eqs. (4.32)-(4.33)."""
        nu_s = 0.
        if alpha_s > 1e-12:
            ratio = ti.pow(self.max_concentration / alpha_s, 1. / 3.) - 1.
            if ratio > 1e-6:
                lamda = 1. / ratio
                nu_s = 1.2 * lamda * lamda * self.water_viscosity * self.density / self.solid_density
        return nu_s

    @ti.func
    def _turbulence_damping(self, alpha_s):
        """[1 - alpha_s / alpha_sm]^n in Eqs. (4.45)-(4.46)."""
        factor = ti.max(1. - alpha_s / self.max_concentration, 0.)
        return ti.pow(factor, self.damping_exponent)

    @ti.func
    def _eddy_viscosity(self, strain_rate_norm, alpha_s):
        """Smagorinsky-Lilly sub-particle scale eddy viscosity, Eqs. (4.45)-(4.48)."""
        length = self.smagorinsky * self.filter_scale
        return length * length * strain_rate_norm * self._turbulence_damping(alpha_s)

    @ti.func
    def _drag_factor(self, relative_velocity_norm, alpha_s):
        """gamma of Eq. (4.37) with Schiller-Naumann (4.38) and Richardson-Zaki (4.39)."""
        reynolds = relative_velocity_norm * self.particle_diameter / self.water_viscosity
        drag_coefficient = 0.44
        if reynolds < 1000.:
            reynolds = ti.max(reynolds, 1e-8)
            drag_coefficient = 24. / reynolds * (1. + 0.15 * ti.pow(reynolds, 0.687))
        alpha_f = ti.max(1. - alpha_s, 1e-6)
        lamda_d = 1. / ti.pow(alpha_f, 1.65)
        return lamda_d * 0.75 * drag_coefficient * self.density / self.particle_diameter * relative_velocity_norm

    @ti.func
    def _deviatoric_free_stress2D(self, velocity_gradient, viscosity):
        """tau*_ij / rho = nu (du_i/dx_j + du_j/dx_i), Eq. (4.50)."""
        stress = ZEROMAT2x2
        stress[0, 0] = 2. * viscosity * velocity_gradient[0, 0]
        stress[1, 1] = 2. * viscosity * velocity_gradient[1, 1]
        stress[0, 1] = viscosity * (velocity_gradient[0, 1] + velocity_gradient[1, 0])
        stress[1, 0] = stress[0, 1]
        return stress

    @ti.func
    def _strain_rate_norm2D(self, velocity_gradient):
        """|S| = sqrt(2 S_ij S_ij), Eq. (4.47)."""
        s00 = velocity_gradient[0, 0]
        s11 = velocity_gradient[1, 1]
        s01 = 0.5 * (velocity_gradient[0, 1] + velocity_gradient[1, 0])
        return ti.sqrt(2. * (s00 * s00 + s11 * s11 + 2. * s01 * s01))
