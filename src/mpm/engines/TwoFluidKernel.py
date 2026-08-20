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
            drift = alpha_s * (velocity_s - velocity) if alpha_s > TRACE_CONCENTRATION else ZEROVEC2f
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
                node[nodeID, bodyID].flux += nvol * drift


@ti.kernel
def kernel_twofluid_grid_velocity(cutoff: float, node: ti.template()):
    for ng, nb in node:
        if node[ng, nb].m > cutoff:
            node[ng, nb].momentum /= node[ng, nb].m
            node[ng, nb].pressure /= node[ng, nb].vol
            node[ng, nb].alpha_s /= node[ng, nb].vol
            node[ng, nb].flux /= node[ng, nb].vol
            # the sediment velocity is only meaningful where sediment is actually
            # present; elsewhere it is slaved to the water to avoid amplifying noise
            if node[ng, nb].ms > cutoff and node[ng, nb].alpha_s > TRACE_CONCENTRATION:
                node[ng, nb].momentums /= node[ng, nb].ms
            else:
                node[ng, nb].momentums = node[ng, nb].momentum
                node[ng, nb].flux = ZEROVEC2f


# ========================================================= #
#               Particle to Grid (forces)                   #
# ========================================================= #
@ti.kernel
def kernel_twofluid_force_p2g(total_nodes: int, start_index: int, end_index: int, gravity: ti.types.vector(3, float),
                              node: ti.template(), particle: ti.template(), materialID: ti.template(), matProps: ti.template(),
                              LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    gravity2D = vec2f(gravity[0], gravity[1])
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes

            # ---- gather the nodal fields required by Eqs. (4.65)-(4.68) ----
            grad_alpha = ZEROVEC2f
            divergence_flux = 0.
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                grad_alpha += node[nodeID, bodyID].alpha_s * dshape_fn
                divergence_flux += node[nodeID, bodyID].flux.dot(dshape_fn)
            particle[np].div_flux = divergence_flux
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
            water_viscosity = matProps.water_viscosity + eddy_viscosity
            sediment_viscosity = matProps._sediment_viscosity(alpha_s) + matProps._eddy_viscosity(sediment_strain_rate, alpha_s)
            water_stress = matProps._deviatoric_free_stress2D(velocity_gradient, water_viscosity)
            sediment_stress = matProps._deviatoric_free_stress2D(sediment_gradient, sediment_viscosity)
            diffusivity = eddy_viscosity / matProps.schmidt

            relative_velocity = particle[np].v - particle[np].vs
            drag = matProps._drag_factor(relative_velocity.norm(), alpha_s)
            # momentum exchange per unit mixture volume, Eq. (4.52)
            exchange = -drag * alpha_s * relative_velocity + drag * (diffusivity / alpha_f) * grad_alpha

            mixture_density = afrf + asrs
            pressure += matProps._artificial_pressure(mixture_density, velocity_gradient.trace())

            water_traction = -volume * afrf * water_stress
            sediment_traction = -volume * asrs * sediment_stress
            # weak form of -grad(alpha_k * p): f_I = + sum_p V_p (alpha_k p)_p grad(N_I)
            water_pressure = volume * alpha_f * pressure
            sediment_pressure = volume * alpha_s * pressure
            # the interfacial pressure p * grad(alpha_k) is transferred between the phases so
            # that their sum reproduces the mixture pressure gradient exactly
            interfacial = volume * pressure * grad_alpha

            water_body = particle[np].m * gravity2D + volume * exchange - interfacial
            sediment_body = particle[np].ms * gravity2D - volume * exchange + interfacial

            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                dshape_fn = dshapefn[ln]
                water_force = shape_fn * water_body + water_pressure * dshape_fn + \
                    vec2f(water_traction[0, 0] * dshape_fn[0] + water_traction[0, 1] * dshape_fn[1],
                          water_traction[1, 0] * dshape_fn[0] + water_traction[1, 1] * dshape_fn[1])
                sediment_force = shape_fn * sediment_body + sediment_pressure * dshape_fn + \
                    vec2f(sediment_traction[0, 0] * dshape_fn[0] + sediment_traction[0, 1] * dshape_fn[1],
                          sediment_traction[1, 0] * dshape_fn[0] + sediment_traction[1, 1] * dshape_fn[1])
                node[nodeID, bodyID].force += water_force
                node[nodeID, bodyID].forces += sediment_force


@ti.kernel
def kernel_twofluid_grid_kinematic(cutoff: float, damp: float, node: ti.template(), dt: ti.template()):
    for ng, nb in node:
        if node[ng, nb].m > cutoff:
            acceleration = node[ng, nb].force / node[ng, nb].m
            node[ng, nb].momentum += acceleration * dt[None]
            node[ng, nb].force = acceleration
            if node[ng, nb].ms > cutoff:
                acceleration_s = node[ng, nb].forces / node[ng, nb].ms
                node[ng, nb].momentums += acceleration_s * dt[None]
                node[ng, nb].forces = acceleration_s
            else:
                node[ng, nb].momentums = node[ng, nb].momentum
                node[ng, nb].forces = acceleration


# ========================================================= #
#                  Grid to Particle (G2P)                   #
# ========================================================= #
@ti.kernel
def kernel_twofluid_g2p(total_nodes: int, alpha: float, dt: ti.template(), start_index: int, end_index: int,
                        node: ti.template(), particle: ti.template(), materialID: ti.template(), matProps: ti.template(),
                        stateVars: ti.template(), LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
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
                sediment_gradient += _outer2D(gvs, dshape_fn)
            particle[np].velocity_gradient = velocity_gradient
            particle[np].sediment_velocity_gradient = sediment_gradient

            water_velocity = particle[np].v
            sediment_velocity = particle[np].vs
            particle[np]._update_particle_state(dt, alpha, vPIC, aFLIP)
            # Eq. (4.68): the material points travel with the water, hence the sediment
            # momentum equation retains the relative advection term
            relative_velocity = sediment_velocity - water_velocity
            advection = sediment_gradient @ relative_velocity
            particle[np].vs = alpha * vPICs + (1. - alpha) * (sediment_velocity + aFLIPs * dt[None]) - advection * dt[None]

            # Eqs. (4.66)-(4.67): mass conservation of both phases.  Equation (4.67) is
            # recast in the Lagrangian form that follows the water:
            #   D(alpha_s)/Dt = -alpha_s div(u_s) - (u_s - u_f) . grad(alpha_s)
            divergence = velocity_gradient.trace()
            jacobian = 1. + divergence * dt[None]
            particle[np].vol *= jacobian
            particle[np].afrf /= jacobian
            alpha_s = particle[np].alpha_s / jacobian - particle[np].div_flux * dt[None]
            # a vanishingly small background concentration keeps the sediment velocity
            # field (and therefore the concentration front) well defined in clear water
            alpha_s = ti.min(ti.max(alpha_s, matProps.background_concentration), MAX_PACKING_FRACTION * matProps.max_concentration)
            particle[np].alpha_s = alpha_s
            particle[np].ms = alpha_s * matProps.solid_density * particle[np].vol
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
