import taichi as ti

from src.utils.constants import ZEROVEC2f, ZEROMAT2x2
from src.utils.TypeDefination import vec2f
from src.utils.ShapeFunctions import *


@ti.func
def _outer2D(a, b):
    return ti.Matrix([[a[0] * b[0], a[0] * b[1]], [a[1] * b[0], a[1] * b[1]]], float)


# ========================================================= #
#                       Grid reset                          #
# ========================================================= #
# below this volumetric concentration a material point is treated as clear water
TRACE_CONCENTRATION = 1e-6
# the sediment viscosity of Eq. (4.32) diverges at the packing limit; keeping the
# concentration slightly below it bounds the viscosity and hence the diffusive stability limit
MAX_PACKING_FRACTION = 0.98


@ti.kernel
def kernel_twofluid_grid_reset(node: ti.template()):
    for ng, nb in node:
        node[ng, nb]._grid_reset()


# ========================================================= #
#             Particle to Grid (kinematics)                 #
# ========================================================= #
@ti.kernel
def kernel_twofluid_mass_momentum_p2g(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(),
                                      LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            mass = particle[np].m
            mass_s = particle[np].ms
            volume = particle[np].vol
            velocity = particle[np].v
            velocity_s = particle[np].vs
            pressure = particle[np].pressure
            alpha_s = particle[np].alpha_s
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                nmass = shape_fn * mass
                nmass_s = shape_fn * mass_s
                nvol = shape_fn * volume
                node[nodeID, bodyID].m += nmass
                node[nodeID, bodyID].momentum += nmass * velocity
                node[nodeID, bodyID].ms += nmass_s
                node[nodeID, bodyID].momentums += nmass_s * velocity_s
                node[nodeID, bodyID].vol += nvol
                node[nodeID, bodyID].pressure += nvol * pressure
                node[nodeID, bodyID].alpha_s += nvol * alpha_s


@ti.kernel
def kernel_twofluid_grid_velocity(cutoff: float, node: ti.template()):
    for ng, nb in node:
        if node[ng, nb].m > cutoff:
            node[ng, nb].momentum /= node[ng, nb].m
            node[ng, nb].pressure /= node[ng, nb].vol
            node[ng, nb].alpha_s /= node[ng, nb].vol
            # the sediment velocity is only meaningful where sediment is actually
            # present; elsewhere it is slaved to the water to avoid amplifying noise
            if node[ng, nb].ms > cutoff and node[ng, nb].alpha_s > TRACE_CONCENTRATION:
                node[ng, nb].momentums /= node[ng, nb].ms
            else:
                node[ng, nb].momentums = node[ng, nb].momentum


# ========================================================= #
#               Particle to Grid (forces)                   #
# ========================================================= #
@ti.kernel
def kernel_twofluid_force_p2g(total_nodes: int, start_index: int, end_index: int, gravity: ti.types.vector(3, float),
                              node: ti.template(), particle: ti.template(), materialID: ti.template(), matProps: ti.template(),
                              LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), node_size: ti.template(),
                              dt: ti.template()):
    gravity2D = vec2f(gravity[0], gravity[1])
    dtime = dt[None]
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes

            # ---- gather the nodal fields required by Eqs. (4.65)-(4.68) ----
            grad_alpha = ZEROVEC2f
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                grad_alpha += node[nodeID, bodyID].alpha_s * dshape_fn
            velocity_gradient = particle[np].velocity_gradient
            sediment_gradient = particle[np].sediment_velocity_gradient

            # ---- closure relations ----
            volume = particle[np].vol
            alpha_s = particle[np].alpha_s
            alpha_f = ti.max(1. - alpha_s, 1e-6)
            afrf = particle[np].afrf
            asrs = alpha_s * matProps.solid_density
            pressure = particle[np].pressure

            water_strain_rate = matProps._strain_rate_norm2D(velocity_gradient)
            sediment_strain_rate = matProps._strain_rate_norm2D(sediment_gradient)
            eddy_viscosity = matProps._eddy_viscosity(water_strain_rate, alpha_s)
            water_viscosity = matProps._stable_viscosity(matProps.water_viscosity + eddy_viscosity, dtime)
            sediment_viscosity = matProps._stable_viscosity(
                matProps._sediment_viscosity(alpha_s) + matProps._eddy_viscosity(sediment_strain_rate, alpha_s), dtime)
            water_stress = matProps._deviatoric_free_stress2D(velocity_gradient, water_viscosity)
            sediment_stress = matProps._deviatoric_free_stress2D(sediment_gradient, sediment_viscosity)
            diffusivity = eddy_viscosity / matProps.schmidt

            relative_velocity = particle[np].v - particle[np].vs
            drag = matProps._drag_factor(relative_velocity.norm(), alpha_s)
            # Momentum exchange per unit mixture volume, Eq. (4.52).  Only the turbulent drift
            # part gamma D/alpha_f grad(alpha_s) is assembled explicitly; the drag proper
            # -gamma alpha_s (u_f - u_s) is a stiff relaxation whose rate gamma alpha_s
            # (1/(alpha_f rho_f) + 1/(alpha_s rho_s)) reaches several 10^3 s^-1 in the dense
            # cloud, so it is integrated implicitly on the grid, see
            # kernel_twofluid_grid_kinematic, which removes it from the time step restriction.
            drift = drag * (diffusivity / alpha_f) * grad_alpha
            drag_coefficient = volume * drag * alpha_s

            mixture_density = afrf + asrs
            pressure += matProps._artificial_pressure(mixture_density, velocity_gradient.trace())

            water_traction = -volume * afrf * water_stress
            sediment_traction = -volume * asrs * sediment_stress
            # The pressure force of the mixture is assembled in the weak form
            # f_I = + sum_p V_p p_p grad(N_I) and is shared by the two phases in proportion to
            # their nodal volume fractions, see kernel_twofluid_grid_kinematic.  Splitting the
            # already assembled mixture force avoids evaluating the two large and almost
            # cancelling contributions -grad(alpha_k p) and p grad(alpha_k) separately, which
            # near a sharp concentration front produces spurious forces of the order of p/Delta.
            mixture_pressure = volume * pressure

            water_body = particle[np].m * gravity2D + volume * drift
            sediment_body = particle[np].ms * gravity2D - volume * drift
            # Eq. (4.67): weak form of the drift flux alpha_s rho_s (u_s - u_f).  Assembling the
            # sediment mass rate on the grid and redistributing it with the same weights makes the
            # concentration transport discretely conservative, because sum_I grad(N_I) vanishes.
            sediment_flux = -volume * asrs * relative_velocity

            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                dshape_fn = dshapefn[ln]
                water_force = shape_fn * water_body + \
                    vec2f(water_traction[0, 0] * dshape_fn[0] + water_traction[0, 1] * dshape_fn[1],
                          water_traction[1, 0] * dshape_fn[0] + water_traction[1, 1] * dshape_fn[1])
                sediment_force = shape_fn * sediment_body + \
                    vec2f(sediment_traction[0, 0] * dshape_fn[0] + sediment_traction[0, 1] * dshape_fn[1],
                          sediment_traction[1, 0] * dshape_fn[0] + sediment_traction[1, 1] * dshape_fn[1])
                node[nodeID, bodyID].force += water_force
                node[nodeID, bodyID].forces += sediment_force
                node[nodeID, bodyID].pforce += mixture_pressure * dshape_fn
                node[nodeID, bodyID].drag += shape_fn * drag_coefficient
                node[nodeID, bodyID].dms += sediment_flux.dot(dshape_fn)


@ti.kernel
def kernel_twofluid_grid_kinematic(cutoff: float, damp: float, node: ti.template(), dt: ti.template()):
    for ng, nb in node:
        if node[ng, nb].m > cutoff:
            concentration = ti.min(ti.max(node[ng, nb].alpha_s, 0.), 1.)
            node[ng, nb].force += (1. - concentration) * node[ng, nb].pforce
            node[ng, nb].forces += concentration * node[ng, nb].pforce
            water_velocity = node[ng, nb].momentum
            sediment_velocity = node[ng, nb].momentums
            velocity = water_velocity + node[ng, nb].force / node[ng, nb].m * dt[None]
            velocitys = sediment_velocity
            if node[ng, nb].ms > cutoff:
                velocitys += node[ng, nb].forces / node[ng, nb].ms * dt[None]
                # implicit relaxation of the drag of Eq. (4.52): the impulse is evaluated with
                # the end-of-step slip, r^{n+1} = r* / (1 + dt K (1/m_f + 1/m_s)), which is
                # unconditionally stable and exchanges momentum exactly between the phases
                mobility = 1. / node[ng, nb].m + 1. / node[ng, nb].ms
                impulse = dt[None] * node[ng, nb].drag / (1. + dt[None] * node[ng, nb].drag * mobility) * \
                    (velocity - velocitys)
                velocity -= impulse / node[ng, nb].m
                velocitys += impulse / node[ng, nb].ms
            else:
                velocitys = velocity
            # the accelerations handed to the FLIP update are the total ones, so that the
            # implicit drag impulse is felt by the material points as well
            node[ng, nb].momentum = velocity
            node[ng, nb].force = (velocity - water_velocity) / dt[None]
            node[ng, nb].momentums = velocitys
            node[ng, nb].forces = (velocitys - sediment_velocity) / dt[None]


# ========================================================= #
#                  Grid to Particle (G2P)                   #
# ========================================================= #
@ti.kernel
def kernel_twofluid_g2p(total_nodes: int, alpha: float, dt: ti.template(), start_index: int, end_index: int,
                        node: ti.template(), particle: ti.template(), materialID: ti.template(), matProps: ti.template(),
                        stateVars: ti.template(), LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(),
                        node_size: ti.template(), correction: ti.template()):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            vPIC, aFLIP = ZEROVEC2f, ZEROVEC2f
            vPICs, aFLIPs = ZEROVEC2f, ZEROVEC2f
            # the velocity gradients are evaluated from the *updated* nodal velocities so
            # that velocity and density are advanced in a staggered (symplectic) manner
            velocity_gradient = ZEROMAT2x2
            sediment_gradient = ZEROMAT2x2
            sediment_mass_rate = 0.
            # the sediment phase only exists where sediment is present; its velocity gradient is
            # therefore evaluated from the sediment-carrying nodes alone and relative to the
            # particle velocity, so that clear water does not act as an artificial rigid support
            # for the cloud through the (very large) Ahilan-Sleath viscosity
            sediment_velocity = particle[np].vs
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                dshape_fn = dshapefn[ln]
                gv = node[nodeID, bodyID].momentum
                gvs = node[nodeID, bodyID].momentums
                vPIC += shape_fn * gv
                aFLIP += shape_fn * node[nodeID, bodyID].force
                vPICs += shape_fn * gvs
                aFLIPs += shape_fn * node[nodeID, bodyID].forces
                velocity_gradient += _outer2D(gv, dshape_fn)
                if node[nodeID, bodyID].alpha_s > TRACE_CONCENTRATION:
                    sediment_gradient += _outer2D(gvs - sediment_velocity, dshape_fn)
                if node[nodeID, bodyID].vol > 0.:
                    sediment_mass_rate += shape_fn * node[nodeID, bodyID].dms / node[nodeID, bodyID].vol
            particle[np].velocity_gradient = velocity_gradient
            particle[np].sediment_velocity_gradient = sediment_gradient

            water_velocity = particle[np].v
            particle[np]._update_particle_state(dt, alpha, vPIC, aFLIP)
            # Eq. (4.68): the material points travel with the water, hence the sediment
            # momentum equation retains the relative advection term
            relative_velocity = sediment_velocity - water_velocity
            advection = sediment_gradient @ relative_velocity
            particle[np].vs = alpha * vPICs + (1. - alpha) * (sediment_velocity + aFLIPs * dt[None]) - advection * dt[None]

            # Eqs. (4.66)-(4.67): mass conservation of both phases.  Equation (4.67) is
            # integrated in its conservative form, D(m_s)/Dt = -V div[alpha_s rho_s (u_s - u_f)],
            # with the divergence assembled on the grid, so that the total sediment mass is
            # conserved to machine precision as long as no bound has to be enforced.
            divergence = velocity_gradient.trace()
            jacobian = 1. + divergence * dt[None]
            # the mass increment is weighted with the volume used in the particle-to-grid
            # assembly, which is what makes the redistribution exactly conservative
            sediment_mass = particle[np].ms + particle[np].vol * sediment_mass_rate * dt[None]
            particle[np].vol *= jacobian
            particle[np].afrf /= jacobian
            volume = particle[np].vol
            # a vanishingly small background concentration keeps the sediment velocity
            # field (and therefore the concentration front) well defined in clear water
            minimum_mass = matProps.background_concentration * matProps.solid_density * volume
            maximum_mass = MAX_PACKING_FRACTION * matProps.max_concentration * matProps.solid_density * volume
            bounded_mass = ti.min(ti.max(sediment_mass, minimum_mass), maximum_mass)
            # the mass artificially created (or destroyed) by enforcing the bounds is
            # book-kept and removed afterwards, see kernel_twofluid_conservative_clip
            correction[0] += bounded_mass - sediment_mass
            correction[1] += bounded_mass
            sediment_mass = bounded_mass
            alpha_s = sediment_mass / (matProps.solid_density * volume)
            particle[np].alpha_s = alpha_s
            particle[np].ms = sediment_mass
            if alpha_s < TRACE_CONCENTRATION:
                particle[np].vs = particle[np].v
            particle[np].pressure = matProps._pressure(particle[np].afrf, alpha_s)
            stateVars[np].pressure = particle[np].pressure
            stateVars[np].concentration = alpha_s


# ========================================================= #
#              Initial hydrostatic equilibrium              #
# ========================================================= #
@ti.kernel
def kernel_twofluid_hydrostatic_state(start_index: int, end_index: int, particle: ti.template(),
                                      materialID: ti.template(), matProps: ti.template(), stateVars: ti.template()):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            pressure = particle[np].pressure
            if pressure > 0.:
                jacobian = matProps._compression_ratio(pressure, particle[np].alpha_s)
                particle[np].vol *= jacobian
                particle[np].afrf /= jacobian
                particle[np].alpha_s /= jacobian
                particle[np].pressure = matProps._pressure(particle[np].afrf, particle[np].alpha_s)
            stateVars[np].pressure = particle[np].pressure
            stateVars[np].concentration = particle[np].alpha_s


# ========================================================= #
#                  Conservative clipping                    #
# ========================================================= #
@ti.kernel
def kernel_twofluid_reset_correction(correction: ti.template()):
    correction[0] = 0.
    correction[1] = 0.


@ti.kernel
def kernel_twofluid_conservative_clip(start_index: int, end_index: int, particle: ti.template(), materialID: ti.template(),
                                      matProps: ti.template(), stateVars: ti.template(), correction: ti.template()):
    """Remove the sediment mass created by the boundedness limiter.

    Clipping the concentration to [alpha_bg, alpha_max] is unavoidable with an explicit
    advection scheme, but it is not conservative.  The spurious mass is redistributed over
    all sediment-carrying material points in proportion to their own mass, which restores
    global conservation without altering the shape of the concentration field.
    """
    scale = 1.
    if correction[1] > 0.:
        scale = (correction[1] - correction[0]) / correction[1]
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            sediment_mass = particle[np].ms * scale
            alpha_s = sediment_mass / (matProps.solid_density * particle[np].vol)
            particle[np].ms = sediment_mass
            particle[np].alpha_s = alpha_s
            if alpha_s < TRACE_CONCENTRATION:
                particle[np].vs = particle[np].v
            particle[np].pressure = matProps._pressure(particle[np].afrf, alpha_s)
            stateVars[np].pressure = particle[np].pressure
            stateVars[np].concentration = alpha_s
