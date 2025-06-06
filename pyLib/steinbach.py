import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt

from phase_space_characterization import characterize_phase_space_at_septum

def steinbach_plot(line, px_gen=0.0, d_gen=0.0, num_turns=1000):
    
    # Define empty arrays for storing the tunes and calculated stable areas
    stable_areas = []
    test_tunes = []
    
    # Run a loop for a range of tunes
    for tune in range(660, 671):
        
        tune = 1 + tune*1e-3
    
        # Match the optics for each tune
        opt = line.match(
            solve=False,
            method='4d',
            vary=[
                xt.VaryList(['s0', 's1'], step=1e-3, tag = 'sext'),
                xt.VaryList(['kf','kd','kr'], limits=(0, 1),  step=1e-3, tag='quad'),
            ],
            targets=[
                xt.TargetSet(dqx=-1, dqy=-1, tol=1e-3, tag="chrom"),
                xt.Target(dx = 0, at='symp', tol=1e-6),
                xt.TargetSet(qx=tune, qy=1.74, tol=1e-6),
            ]
        )
        opt.step(20)      
    
        # Call the function for calculating quantities without plotting phase space
        # and store caclulated quantities on a dictionary (we only use stable area)
        values = characterize_phase_space_at_septum(line, x_gen=0, px_gen=px_gen, d_gen=d_gen, num_turns=num_turns)
    
        # Store stable areas and tunes
        stable_areas.append(values['stable_area'])
        test_tunes.append(tune)    
            
        tune = (tune - 1) * 1e3

    roots = np.sqrt(stable_areas)

    # We find the first degree polynomial fit coefficients
    poly_1 = np.polyfit(test_tunes[:7], roots[:7], deg=1)
    poly_2 = np.polyfit(test_tunes[7:11], roots[7:11], deg=1)
    
    # Calculate the resonace value (zero stable area) according to the polyfit coefficients
    polyres1 = -poly_1[1] / poly_1[0]
    polyres2 = -poly_2[1] / poly_2[0]

    plt.figure(figsize=(10, 5))
    st_plot = plt.subplot(1, 1, 1, title='Steinbach Diagram')
    st_plot.set_xlabel(r'$Q_x$')
    st_plot.set_ylabel(r'$\sqrt{\text{Stable Areas}}$')
    st_plot.grid(True)
    st_plot.axhline(y=0, color='k', alpha=0.8)
    
    # Create the extended lines (fitting)
    x_plt1 = [1.660, polyres1]
    st_plot.plot(x_plt1, np.polyval(poly_1, x_plt1), '-', color='#1f77b4')
    
    x_plt2 = [polyres2, 1.6725]
    st_plot.plot(x_plt2, np.polyval(poly_2, x_plt2), '-', color='#1f77b4')

    return st_plot

    