import numpy as np
import matplotlib.pyplot as plt
import json

from scipy.constants import c as clight
from scipy import signal

import xtrack as xt
import xobjects as xo

line = xt.Line.from_json('../PIMMStutorials_CNAOlattice/cnao_lattice_00_optics.json')
line.build_tracker()


### Define reference particle ###
line.particle_ref = xt.Particles(q0=1, mass0=xt.PROTON_MASS_EV,
                                 kinetic_energy0=60e6)


### Match ###

# Build a single match with all constraints (can be reused to retune the machine)
opt = line.match(
    solve=False,
    method='4d',
    vary=[
        xt.VaryList(['s0', 's1'], step=1e-3, tag = 'sext'),
        xt.VaryList(['kf', 'kd','kr'], limits=(0, 1),  step=1e-3, tag='quad')
    ],
    targets=[
        xt.TargetSet(dqx=-4, dqy=-1, tol=1e-3, tag="chrom"),
        xt.Target(dx = 0, at='symp', tol=1e-6),
        xt.TargetSet(qx=1.666, qy=1.74, tol=1e-6),
    ]
)
# Perform twenty optimization steps
opt.step(20)


### Twiss ###
tw = line.twiss(method='4d')


### Build a matched beam distribution ###

# Define number of turns and number of particles for the simulation
num_turns = 100000
num_particles = 5000

# Generate Gaussian distribution in normalized phase space
x_norm = np.random.normal(size=num_particles)
px_norm = np.random.normal(size=num_particles)
y_norm = np.random.normal(size=num_particles)
py_norm = np.random.normal(size=num_particles)

# Generate Gaussian momentum offset distribution (mean 6e-4, std spread 1e-4)
delta = np.random.normal(loc=6e-4, scale=1e-4, size=num_particles)

# Particles arrival time spread over one turn
zeta = np.random.uniform(size=num_particles) * line.get_length()

# Assemble Particles object
particles = line.build_particles(
    x_norm=x_norm, px_norm=px_norm, 
    y_norm=y_norm, py_norm=py_norm,
    delta=delta,
    zeta=zeta,
    method='4d',
    nemitt_x=0.52e-6, nemitt_y=1e-8,
)

# save initial state
p0 = particles.copy()


### Define time-dependent bevior of extraction sextupole ###

line.functions['fun_xsext'] = xt.FunctionPieceWiseLinear(x=[0, 0.5e-3, 3e-3], y=[0, 0, 1.])
line.vars['sr'] *= line.functions['fun_xsext'](line.vars['t_turn_s'])


### Set realistic septum aperture ###
line['septum_aperture'].max_x = 0.035


### Switch to multithreaded context to gain speed ###
line.discard_tracker()
ctx = xo.ContextCpu(omp_num_threads='auto')  # 'auto' for multithread | 0 for single thread
line.build_tracker(_context=ctx)


### Define quantities to be logged during tracking ###

# User-defined quantity to be logged (functions executed at every turn, output is automatically logged).
def measure_intensity(line, particles):
    intensity = np.sum(particles.state)/num_particles
    return intensity

log = xt.Log('sr',                         # vars to be logged
             intensity=measure_intensity)  # user-defined function to be logged


### Enable time-dependent vars update for tracking ###
line.enable_time_dependent_vars = True


## Track! ###
line.track(particles, num_turns=num_turns, with_progress=True, log=log)


## Save plots with logged quantities ###
plt.figure()

ax1 = plt.subplot(2,1,1)
plt.plot(line.log_last_track['sr'], label='sr')
plt.ylabel('Sext. strength [$m^{-2}$]')
plt.legend()

ax2 = plt.subplot(2,1,2, sharex=ax1)
plt.plot(line.log_last_track['intensity'])
plt.ylim(bottom=0)
plt.ylabel('Intensity [$p^+$]')

plt.savefig("Sextupole_ramp_intensity.png")

# Alive percentage after the tracking only with the resonance sextupole ramp-up
intensity_only_sextupole = line.log_last_track["intensity"][-1]


### Plot phase space after tracking ###
mask_alive = particles.state>0
mask_lost = ~mask_alive

#Get particles normalized coordinates
part_nc = tw.get_normalized_coordinates(particles)

plt.figure(figsize=(10, 5))
ax_geom = plt.subplot(1, 2, 1, title='Physical phase space')
ax_norm = plt.subplot(1, 2, 2, title='Normalized phase space')

ax_geom.plot(particles.x[mask_alive], particles.px[mask_alive],
         '.', markersize=1, color='green', label='circulating')
ax_geom.plot(particles.x[mask_lost], particles.px[mask_lost],
         '.', markersize=1, color='red', label='extracted')
ax_geom.set_xlim(-0.03, 0.05); ax_geom.set_ylim(-3e-3, 3e-3)
ax_geom.axvline(x=line['septum_aperture'].max_x, color='k', ls='--')
ax_geom.set_xlabel(r'${x} [m]$')
ax_geom.set_ylabel(r'${p}_x$')
ax_geom.legend()

ax_norm.plot(part_nc.x_norm[mask_alive], part_nc.px_norm[mask_alive],
         '.', markersize=1, color='green', label='circulating')
ax_norm.plot(part_nc.x_norm[mask_lost], part_nc.px_norm[mask_lost],
         '.', markersize=1, color='red', label='extracted')
ax_norm.set_aspect('equal', adjustable='datalim')
ax_norm.set_xlabel(r'$\hat{x}$')
ax_norm.set_ylabel(r'$\hat{p}_x$')
ax_norm.legend()

plt.tight_layout()
plt.savefig("Phase_space_after_tracking_Sextupole_ramp.png")


### Introduce transverse excitation to control the spill ###
#We build a custom beam element that excites the beam with a sinusoidal function of time

# Define new element type
class RFKOExciter:
    def __init__(self):
        
        self.amplitude = 0
        self.tune = 0
        self.f_rev = 1 / tw.T_rev0

    def track(self, p):

        f_excit = self.tune * self.f_rev

        # Time of arrival corrected for delay within the turn
        t_particle_pass = (p.at_turn[p.state > 0] / self.f_rev
                              - p.zeta[p.state > 0] / p.beta0[0] / clight) 

        p.px[p.state > 0] += (self.amplitude * np.sin(2 * np.pi * f_excit * t_particle_pass))
        

# Install the custom element (thin element) in the center of the actual RFKO Exciter.
# This way the actual RFKO Exciter of the CNAO lattice (which is considered a drift) is sliced.
line.discard_tracker()
line.insert_element('rfko_exc', RFKOExciter(), at_s=63.4361)
line.build_tracker(_context=ctx)

# Define a function that activates the exciter amplitude after 6e-3 seconds
# and increase the amplitude linearly until the end of the simulation
line.functions['fun_excit'] = xt.FunctionPieceWiseLinear(
    x=[0, 6e-3,  tw.T_rev0 * num_turns],
    y=[0, 0,     4e-5])

# Create a variable linked to the current turn time
line.vars['ampl_excit'] = line.functions['fun_excit'](line.vars['t_turn_s'])

# Assign the variable to the class amplitude attribute
line.element_refs['rfko_exc'].amplitude = line.vars['ampl_excit']


### Sawtooth ###

# Define frequency range
min_freq = 0.662
max_freq = 0.665

# We create a sawtooth signal for the frequency modulation of the RFKO Exciter
t_start = 6e-3                    # When the sawtooth signal begins
t_end = tw.T_rev0 * num_turns     # When the sawtooth signal ends (end of the simulation)
duration = t_end - t_start

n_points = 2 * num_turns    # Number of samples
n_peaks = 10                # Number of peaks in the signal

# Sawtooth period
T = duration / n_peaks
f_saw = 1 / T               # Frequency in Hz

# Array of time stamps
t = np.linspace(t_start, t_end, n_points)

# Scale sawtooth to [min_freq, max_freq]
fr_range = max_freq - min_freq
fr_mid = min_freq + fr_range/2

# Create the signal samples
saw_values = (fr_range / 2) * signal.sawtooth(2 * np.pi * f_saw * (t - t_start), width=1) + fr_mid

t_full = np.concatenate((
    [0.0, t_start-1e-9],     # just before the waveform
    t,
    [t_end+1e-9]             # just after
))

y_full = np.concatenate((
    [min_freq, min_freq],    # constant before start
    saw_values,
    [saw_values[-1]]         # hold last value
))


# Define a function to assign the sawtooth signal to the exciter frequency
line.functions['fun_saw_freq'] = xt.FunctionPieceWiseLinear(x=t_full, y=y_full)

# Create a variable linked to the current turn time
line.vars['freq_excit'] = line.functions['fun_saw_freq'](line.vars['t_turn_s'])

# Assign the variable to the class tune attribute
line.element_refs['rfko_exc'].tune = line.vars['freq_excit']

# Reset simulation time
line.vars['t_turn_s'] = 0

# Back to initial particles distribution
p = p0.copy()

# Log excitation parameters
log = xt.Log(
    'sr', 't_turn_s',              # vars to be logged
    'ampl_excit', 'freq_excit',
    intensity=measure_intensity)   # user-defined functions to be logged


### Track! ###
line.track(p, num_turns=num_turns, with_progress=True, log=log)


### Plot logged quantities ###
plt.figure(figsize=(10, 7))

t_ms = np.array(line.log_last_track['t_turn_s']) * 1e3

ax1 = plt.subplot(4,2,1)
plt.plot(t_ms, line.log_last_track['sr'], label='sr')
plt.ylabel('Sext. strength [$m^{-2}$]')
plt.legend()
plt.grid()

ax2 = plt.subplot(4,2,2)
plt.plot(line.log_last_track['sr'], label='sr')
plt.legend()
plt.grid()

ax3 = plt.subplot(4,2,3, sharex=ax1)
plt.plot(t_ms, line.log_last_track['intensity'])
plt.ylim(bottom=0)
plt.ylabel('Intensity [$p^+$]')
plt.grid()

ax4 = plt.subplot(4,2,4, sharex=ax2)
plt.plot(line.log_last_track['intensity'])
plt.ylim(bottom=0)
plt.grid()

ax5 = plt.subplot(4,2,5, sharex=ax1)
plt.plot(t_ms, line.log_last_track['ampl_excit'])
plt.ylim(bottom=0)
plt.ylabel('Excit. ampl.')
plt.grid()

ax6 = plt.subplot(4,2,6, sharex=ax2)
plt.plot(line.log_last_track['ampl_excit'])
plt.ylim(bottom=0)
plt.grid()

ax7 = plt.subplot(4,2,7, sharex=ax1)
plt.plot(t_ms, line.log_last_track['freq_excit'])
plt.ylabel('Excit. freq.')
plt.xlabel('Time [ms]')
plt.grid()

ax8 = plt.subplot(4,2,8, sharex=ax2)
plt.plot(line.log_last_track['freq_excit'])
plt.xlabel('Turn')
plt.grid()

plt.tight_layout()
plt.savefig("RFKO_logged_quantities.png")

# Alive percentage
intensity_with_rfko_exciter = line.log_last_track["intensity"][-1]


### Plot phase space after tracking ###
mask_alive = p.state>0
mask_lost = ~mask_alive

# Get particles normalized coordinates
p_nc = tw.get_normalized_coordinates(p)

plt.figure(figsize=(10, 5))
ax_g = plt.subplot(1, 2, 1, title='Physical phase space')
ax_n = plt.subplot(1, 2, 2, title='Normalized phase space')

ax_g.plot(p.x[mask_alive], p.px[mask_alive],
         '.', markersize=1, color='green', label='circulating')
ax_g.plot(p.x[mask_lost], p.px[mask_lost],
         '.', markersize=1, color='red', label='extracted')
ax_g.set_xlim(-0.03, 0.05); ax_g.set_ylim(-3e-3, 3e-3)
ax_g.axvline(x=line['septum_aperture'].max_x, color='k', ls='--')
ax_g.set_xlabel(r'${x} [m]$')
ax_g.set_ylabel(r'${p}_x$')
ax_g.legend()

ax_n.plot(p_nc.x_norm[mask_alive], p_nc.px_norm[mask_alive],
         '.', markersize=1, color='green', label='circulating')
ax_n.plot(p_nc.x_norm[mask_lost], p_nc.px_norm[mask_lost],
         '.', markersize=1, color='red', label='extracted')
ax_n.set_aspect('equal', adjustable='datalim')
ax_n.set_xlabel(r'$\hat{x}$')
ax_n.set_ylabel(r'$\hat{p}_x$')
ax_n.legend()

plt.tight_layout()
plt.savefig("Phase_space_after_tracking_with_RFKO.png")

# Extracted particles' px spread at electrostatic septum in mrad
px_spread_in_mrad = (p.px[mask_lost].max() - p.px[mask_lost].min()) * 1e3


# Save last turn particle data in json file
particles_dict = {key: value.tolist() for key, value in p.to_dict().items()}

simulation_data = {
    "intensity after sextupole": intensity_only_sextupole,
    "intensity after RFKO": intensity_with_rfko_exciter, 
    "px spread after RFKO": px_spread_in_mrad,
    "tracking time": line.time_last_track,
    "logged quantities after RFKO": line.log_last_track,
    "particles after tracking": particles_dict    
}

with open("simulation_data.json", "w") as f:
    json.dump(simulation_data, f, indent=2)


