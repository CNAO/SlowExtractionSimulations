"""
This a basic python module for managing the lattice and optics, e.g.
   inspecting tune, chroma, twiss functions, phase spaces, etc...
"""


__version__ = '0.0'
__author__ = 'Angelos Efstratiou and Alessio Mereghetti'

def FindClosestStableAndUnstable(myLine, num_turns=1000, d_gen=0.0, xBoundary=3.5e-2, xSearch=[0, 1], absPrecisio=1e-6):
    """
    This function allows to identify the two closest particles in the
	stable and unstable area, around xBoundary.
    xSearch denotes the search interval on the horizontal axis.	
    
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
    
