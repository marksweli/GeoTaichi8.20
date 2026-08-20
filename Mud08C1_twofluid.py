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

    # ---- post-processing following Shi Huabin, section 5.3 -------------------
    # Z  : drop distance of the cloud centre (sediment-mass weighted centroid)
    # w_c: sinking velocity of the cloud centre, normalised by omega_s
    # B  : cloud width, bounded by the points where alpha_s = 0.1 alpha_s^max
    omega_s = 0.1545
    L0 = np.sqrt(5e-4)
    fs = sorted(glob.glob('mud_twofluid_08C1/particles/MPMParticle*'))
    print(f'Frames:{len(fs)}')
    history = []
    for f in fs:
        d = np.load(f, allow_pickle=True)
        t = float(d['t_current'])
        ms = d['sediment_mass']; conc = d['concentration']; pos = d['position']
        total = float(ms.sum())
        if total <= 0.:
            continue
        yc = float((ms * pos[:, 1]).sum() / total)
        cloud = conc >= 0.1 * conc.max()
        width = float(pos[cloud, 0].max() - pos[cloud, 0].min()) if cloud.any() else 0.
        history.append((t, yc, width, total, float(conc.max()), bool(np.isnan(pos).any())))
    y0 = history[0][1]
    print(' t[s]    Z/L0    w_c/omega_s   B/L0   alpha_s^max   m_s error')
    for i, (t, yc, width, total, cmax, nan) in enumerate(history):
        if nan:
            print(f'{t:6.3f}  diverged'); break
        w = 0.
        if 0 < i < len(history) - 1:
            w = (history[i - 1][1] - history[i + 1][1]) / (history[i + 1][0] - history[i - 1][0])
        elif i > 0:
            w = (history[i - 1][1] - yc) / (t - history[i - 1][0])
        print(f'{t:6.3f} {(y0 - yc) / L0:7.2f} {w / omega_s:12.3f} {width / L0:7.2f} {cmax:12.4f}'
              f' {(total - history[0][3]) / history[0][3]:11.2e}')

if __name__ == '__main__':
    main()
