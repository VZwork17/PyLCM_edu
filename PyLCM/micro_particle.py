import numpy as np
from matplotlib import pyplot as plt

from PyLCM.parameters import *
from PyLCM.parameters import *
from PyLCM.parcel import *
from PyLCM.condensation import *

class particles:
    def __init__(self,n):
        self.id     = 0
        self.M      = 1.0 # mass
        self.A      = 1.0 # weighting factor
        self.Ns     = 1.0 # Aerosol mass
        self.kappa  = 0.5 # kappa parameter
        self.z      = 0.0 # particle vertical location
    def shuffle(particles_list, rng=None):
        """Deterministically shuffle particles_list using passed RNG. If rng is None, creates a local RNG."""
        if rng is None:
            rng = np.random.default_rng()
        if len(particles_list) <= 1:
            return
        # Use rng.permutation to reorder in-place
        perm = rng.permutation(len(particles_list))
        particles_list[:] = [particles_list[i] for i in perm]