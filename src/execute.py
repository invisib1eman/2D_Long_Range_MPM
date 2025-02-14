import numpy as np
import cProfile
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
from mean_field_model import Dipole_model
import random
from direct_sum import Direct_sum_Dipole_model
import pandas as pd
# Convert experimental values to dimensionless variables

# Define constants

# Define simulation parameters

# Particle radius is unit length 50 nm

R = 1

# Calculate the lattice constant of hexagonal lattice

a_list = [4, 6, 8, 9, 10, 11, 12, 13, 14]
# Generate the lattice positions

N = 20*20
position = np.zeros((N,3))
for a in a_list:
    for i in range(20):
        for j in range(20):
            position[i*20+j] = np.array([i*a,j*a*np.sqrt(3)/2,0])

# nondimensionalize the parameters

# wavevectors 

lambda_array = np.linspace(6,14,100)

k_array = 2*np.pi/lambda_array

# polarizability

eps_medium = 1
optical_df = pd.read_csv('optical_data.csv')


# Calculate the dipole moment

# When using direct sum method:
dipole_moment = Direct_sum_Dipole_model(position, alpha, k)  # No box parameter

# When using mean field method (which does use box):
dipole_moment = Dipole_model(position, alpha, box, k)  # Box parameter included







