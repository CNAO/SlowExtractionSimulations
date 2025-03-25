"""
This a basic python module for managing the lattice and optics, e.g.
   inspecting tune, chroma, twiss functions, phase spaces, etc...
"""


__version__ = '0.0'
__author__ = 'Angelos Efstratiou and Alessio Mereghetti'


import numpy as np
import matplotlib.pyplot as plt


def closest_stable_unstable(myLine, num_turns=1000, d_gen=0.0, xBoundary=3.5e-2, xSearch=[0, 1], absPrecisio=1e-6):
    """
    This function allows to identify the two closest particles in the
    stable and unstable area, around xBoundary.
    xSearch denotes the search interval on the horizontal axis.	
    
    return: search region
    
    myLine: an XSuite line object;
    xBoundary, xSearch, absPrecisio: [m];
    """

    while xSearch[1] - xSearch[0] > absPrecisio:
        
        # Generate a particle in the middle of the region
        x_test = (xSearch[0] + xSearch[1]) / 2
        p = myLine.build_particles(x=x_test, delta=d_gen)
        
        # Track
        myLine.track(p, num_turns=num_turns, turn_by_turn_monitor=True)
        rec_test = myLine.record_last_track
        
        # Update the search region
        if (rec_test.x > xBoundary).any():
            # Test particle is unstable
            # => Sepearatrix is on the right w.r.t x_test
            xSearch[1] = x_test
        else:
            # Test particle is stable
            # Sepearatrix is on the left w.r.t x_test
            xSearch[0] = x_test
            
    return xSearch
    
    
    
def find_separatrix(myLine, xSearch, num_turns=1000, d_gen=0.0):
    """
    Track a particle in the outer edge of the narrowed-down search
    region and get its physical coordinates after each turn.
    
    return: record separatrix
    
    myLine: an XSuite line object;
    xSearch: [m];
    """

    # We track particles at the outer edge of narrowed-down search region
    p = myLine.build_particles(x=xSearch[1], delta=d_gen)
    myLine.track(p, num_turns=num_turns, turn_by_turn_monitor=True)

    separatrix = myLine.record_last_track

    return separatrix
    

def find_separatrix_slope(separatrix, xBoundary=3.5e-2):
    """
    This functions finds the turn at which the particle was closer
    to the septum and fits a straight line using the previous and 
    the following passage.
    
    return: slope at separatrix
    
    separatrix: XSuite turn-by-turn data;
    xBoundary: [m];
    """
    
    x_separ = separatrix.x[0, :]
    px_separ = separatrix.px[0, :]

    # Find turn at which the particle was closer to the septum
    i_septum = np.argmin(np.abs(x_separ - xBoundary))
    
    # Fit a straight line using the previous and the following passage
    # (takes three turns to come back to the same branch)
    poly_sep = np.polyfit([x_separ[i_septum + 3], x_separ[i_septum - 3]],
                      [px_separ[i_septum + 3], px_separ[i_septum - 3]],
                       deg=1)
    
    slope_at_separatrix = poly_sep[0]
    
    return slope_at_separatrix
    
   
def px_at_septum(separatrix, xBoundary=3.5e-2):
    """
    This function finds the px value at the 
    electrostatic septum.
    
    return: px at septum
    
    separatrix: XSuite turn-by-turn data;
    xBoundary: [m];
    """
    x_separ = separatrix.x[0, :]
    px_separ = separatrix.px[0, :]
    
    # Find turn at which the particle was closer to the septum
    i_septum = np.argmin(np.abs(x_separ - xBoundary))
    
    # Fit a straight line using the previous and the following passage
    # (takes three turns to come back to the same branch)
    poly_sep = np.polyfit([x_separ[i_septum + 3], x_separ[i_septum - 3]],
                      [px_separ[i_septum + 3], px_separ[i_septum - 3]],
                       deg=1)
                       
    # px where the least square straight line 
    # crosses the electrostatic septum
    px_at_septum = np.polyval(poly_sep, xBoundary)
    
    return px_at_septum
    
    
def find_triangle(myLine, xSearch, num_turns=1000, d_gen=0.0):
    """"
    Track a particle in the inner edge of narrowed-down search
    region and get its physical coordinates after each turn.

    return: record triangle
    
    myLine: an XSuite line object;
    xSearch: [m];
    """
    # We track particles at the inner edge of narrowed-down search region 
    p = myLine.build_particles(x=xSearch[0], delta=d_gen)
    myLine.track(p, num_turns=num_turns, turn_by_turn_monitor=True)
    
    triangle = myLine.record_last_track
    
    return trianlge






