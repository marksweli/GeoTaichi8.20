"""
08C1 water-sediment two-fluid column-collapse reproduction (Shi Huabin, Ch.4/5).

Implements the Eulerian-Eulerian water + suspended-sediment model via the new
`TwoFluidSediment` material type: two momentum equations (water u_f / sediment u_s)
coupled by Schiller-Naumann + Richardson-Zaki drag, mixture equation of state,
Smagorinsky SPS and Ahilan-Sleath sediment viscosity.

Experiment 08C1: dp=0.8 mm, omega_s=15.45 cm/s, q0=5 cm^2, alpha_s0=0.606,
domain 1 m x 1 m, particle/grid spacing Delta=0.005 m.
"""
from geotaichi import *
import numpy as np, glob

def main():
    init(dim=2, arch='cpu', cpu_max_num_threads=16)
    mpm = MPM()

    G = 9.8; h = 0.005; H = 1.0
    rho_f0 = 1000.; rho_s = 2650.
    dp = 0.0008; nu_f0 = 1e-6
    cs_smag = 0.100; sc = 1.0; alpha_sm = 0.65
    c0 = 10.0 * (G * H) ** 0.5          # > 10x max flow velocity (paper rule)
    gamma_prime = 7.; n = 5
    alpha_s0 = 0.606
    npic = 2                            # -> particle spacing Delta = 0.005 m (paper)
    dx = h / npic

    # q0 = 5 cm^2 released at the free surface, aligned with the particle lattice
    mud_width = 8 * dx; mud_height = 10 * dx
    mx0 = 0.5 - 0.5 * mud_width; my0 = H - mud_height

    mpm.set_configuration(domain=[H, 1.1 * H], background_damping=0.0, alphaPIC=0.85,
        mapping='USL', shape_function='QuadBSpline', gravity=[0., -G],
        material_type='TwoFluidSediment', velocity_projection='PIC')
    mpm.set_solver({'Timestep': 2.5e-5, 'SimulationTime': 1.0, 'SaveInterval': 0.05,
                    'SavePath': 'mud_twofluid_08C1'})
    mpm.memory_allocate(memory={'max_material_number': 2, 'max_particle_number': 400000,
        'verlet_distance_multiplier': 1., 'max_constraint_number': {'max_reflection_constraint': 400000}})

    base = {'Density': rho_f0, 'SolidDensity': rho_s, 'ParticleDiameter': dp,
            'WaterKinematicViscosity': nu_f0, 'SmagorinskyCoefficient': cs_smag,
            'SchmidtNumber': sc, 'MaxConcentration': alpha_sm, 'SoundSpeed': c0,
            'TaitExponent': gamma_prime, 'DampingExponent': n, 'ElementLength': h,
            'FilterScale': dx, 'cL': 1.0, 'cQ': 2.0}
    water = dict(base); water.update({'MaterialID': 1, 'Concentration': 0.0})
    mpm.add_material(model='TwoFluidSediment', material=water)

    mpm.add_element({'ElementType': 'Q4N2D', 'ElementSize': [h, h]})
    cs = {'InternalStress': [0., 0., 0., 0., 0., 0.]}

    mpm.add_region([
        {'Name': 'full_water', 'Type': 'Rectangle2D', 'BoundingBoxPoint': [0., 0.], 'BoundingBoxSize': [1., 1.], 'ydirection': [0., 1.]},
        {'Name': 'mud', 'Type': 'Rectangle2D', 'BoundingBoxPoint': [mx0, my0], 'BoundingBoxSize': [mud_width, mud_height], 'ydirection': [0., 1.]}
    ])
    # one single water body; the released mud is prescribed afterwards as a concentration
    # patch so that the material points are never duplicated
    mpm.add_body({'Template': [
        {'RegionName': 'full_water', 'nParticlesPerCell': npic, 'BodyID': 0, 'MaterialID': 1, 'ParticleStress': cs, 'InitialVelocity': [0., 0.], 'FixVelocity': ['Free', 'Free']}
    ]})
    mpm.update_particle_properties(region_name='mud', property_name='concentration', value=alpha_s0)
    mpm.add_boundary_condition(boundary=[
        {'BoundaryType': 'ReflectionConstraint', 'Norm': [0., -1.], 'StartPoint': [0., 0.], 'EndPoint': [1., 0.]},
        {'BoundaryType': 'ReflectionConstraint', 'Norm': [-1., 0.], 'StartPoint': [0., 0.], 'EndPoint': [0., 1.1]},
        {'BoundaryType': 'ReflectionConstraint', 'Norm': [1., 0.], 'StartPoint': [1., 0.], 'EndPoint': [1., 1.1]}
    ])
    mpm.select_save_data(particle=['MaterialID', 'Velocity'])
    mpm.run(gravity_field=True)

    fs = sorted(glob.glob('mud_twofluid_08C1/particles/MPMParticle*'))
    print(f'Frames:{len(fs)}')
    for f in fs[::8]:
        d = np.load(f, allow_pickle=True)
        tc = float(d['t_current'])
        ok = not np.any(np.isnan(d['position']))
        ms = d['sediment_mass']
        conc = d['concentration']
        total_ms = float(ms.sum())
        sed = conc > 0.001
        if ms.sum() > 0:
            my = float((ms * d['position'][:, 1]).sum() / ms.sum())
            xc = float((ms * d['position'][:, 0]).sum() / ms.sum())
            width = float(np.sqrt((ms * (d['position'][:, 0] - xc) ** 2).sum() / ms.sum()))
        else:
            my = xc = width = 0.0
        surf = float(d['position'][:, 1].max())
        print(f't={tc:.3f} ok={ok} mud_y={my:.4f} surf={surf:.4f} width={width:.4f} ms_tot={total_ms:.4e} sed_n={int(sed.sum())}')
    if len(fs) > 1:
        d0 = np.load(fs[0], allow_pickle=True); d1 = np.load(fs[-1], allow_pickle=True)
        m0 = d0['sediment_mass']; m1 = d1['sediment_mass']
        if m0.sum() > 0 and m1.sum() > 0:
            y0 = float((m0 * d0['position'][:, 1]).sum() / m0.sum())
            y1 = float((m1 * d1['position'][:, 1]).sum() / m1.sum())
            ws = (y0 - y1) / float(d1['t_current'])
            print(f'w_s={ws:.4f} m/s (target 0.1545)')
            print(f'sediment mass conservation: {m0.sum():.4e} -> {m1.sum():.4e} (rel err {(m0.sum()-m1.sum())/m0.sum():.2e})')

if __name__ == '__main__':
    main()
