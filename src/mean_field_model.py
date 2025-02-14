import numpy as np
import cProfile
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
import pandas as pd
from numba import jit, prange
import numpy.typing as npt

#Dipole model
#epsilon_medium = 1
#position: N_particles,3, particle positions
#alpha: N_particles, particle polarizability
#box: 3,3, box size
#k: 3, wavevector of the incident light , firstly I assume it is in the z direction k= [0,0,k_z]
#E0: 2,3, electric field of the incident light at two different directions perpendicular to wavevector
def Dipole_model(position, alpha, box, k):
    N_particles = position.shape[0]
    k_hat = k/np.linalg.norm(k)
    # Using [1,0,0] unless k_hat is parallel to it
    temp_vec = [1,0,0] if not np.allclose(np.abs(k_hat), [1,0,0]) else [0,1,0]
    E1 = np.cross(k_hat, temp_vec)
    E1 = E1 / np.linalg.norm(E1)
    
    # Find second perpendicular vector using cross product
    E2 = np.cross(k_hat, E1)
    E2 = E2 / np.linalg.norm(E2)
    E0 = np.array([E1,E2])
    dipole_moment = np.zeros((N_particles,3,2), dtype=complex)
    for i,E in enumerate(E0):
        dipole_moment_guess = np.repeat([alpha*E], N_particles, axis=0)
        dipole_moment_solution = Solve_Dipole(position, dipole_moment_guess, box, k, E, alpha)
        dipole_moment[:,:,i] = dipole_moment_solution
    return dipole_moment
def Solve_Dipole(position, dipole_moment_guess, box, k, E, alpha):
    #Solve the dipole moment using the dipole moment guess
    dipole_moment_solution = dipole_moment_guess
    neighbor_list_here = neighbor_list(position, box)
    n = position.shape[0] * 3  # Multiply by 3 for x,y,z components
    A_operator = LinearOperator((n,n), matvec=make_matvec(alpha, dipole_moment_solution, position, box, k, neighbor_list_here, E),dtype=complex)
    E_array = np.repeat([E], position.shape[0], axis=0)
    E_array_flat = E_array.reshape(-1)
    # Reshape input for GMRES
    flat_guess = dipole_moment_guess.reshape(-1)
    print(flat_guess)
    solution = gmres(A_operator, E_array_flat, rtol=1e-6, x0=flat_guess)[0]
    
    # Reshape solution back to original shape
    dipole_moment_solution = solution.reshape(-1, 3)
    return dipole_moment_solution
def make_matvec(alpha, dipole_moment, position, box, k, neighbor_list, E):
    """Optimized matvec operation"""
    def matvec(x):
        x_reshaped = x.reshape(-1, 3)
        result = -Total_dipole_field(x_reshaped, position, box, k, neighbor_list) + (1/alpha)*x_reshaped
        return result.reshape(-1)
    return matvec
def Total_field(dipole_moment, position, box, k, neighbor_list, E):
    field = Total_dipole_field(dipole_moment, position, box, k, neighbor_list)
    print(f"ratio: {field/E}")
    field += E
    return field
def Total_dipole_field(dipole_moment, position, box, k, neighbor_list):
    """Combined field calculation"""
    field = sum_direct_dipole_field(dipole_moment, position, box, k, neighbor_list)
    field += total_smeared_dipole_field(dipole_moment, position, box, k)
    return field

@jit(nopython=True)
def smeared_dipole_field(dipole_moment, dipole_position, position, box, k):
    """Optimized smeared dipole field calculation"""
    area_9_box = box[0] * box[1] * 9
    radius_9_box = np.sqrt(area_9_box/np.pi)
    k_abs = np.linalg.norm(k)
    delta_z = np.abs(position[2] - dipole_position[2])
    cutoff_r = np.sqrt(delta_z**2 + radius_9_box**2)
    area = box[0] * box[1]
    return 1j * k_abs * dipole_moment * np.exp(1j*k_abs*cutoff_r)/(2*area)

@jit(nopython=True, parallel=True)
def total_smeared_dipole_field(dipole_moment, position, box, k):
    """Parallelized smeared dipole field calculation"""
    field = np.zeros((position.shape[0], 3), dtype=np.complex128)
    for i in prange(position.shape[0]):
        for j in range(position.shape[0]):
            field[i] += smeared_dipole_field(
                dipole_moment[j], 
                position[j], 
                position[i], 
                box, 
                k
            )
    return field

@jit(nopython=True)
def direct_dipole_field(dipole_moment, dipole_position, position, box, k):
    """Optimized dipole field calculation with periodic boundary conditions"""
    delta_r = position - dipole_position
    delta_r_abs = np.linalg.norm(delta_r)
    k_abs = np.linalg.norm(k)
    
    # Pre-calculate common terms
    exp_term = np.exp(1j * k_abs * delta_r_abs)
    dipole_dot_r = np.dot(dipole_moment, delta_r)
    common_term = (delta_r_abs**2 * dipole_moment - 3 * delta_r * dipole_dot_r)
    
    # Calculate terms
    long_range = -k_abs**2 * exp_term * (delta_r * dipole_dot_r - dipole_moment * delta_r_abs**2) / delta_r_abs**3
    mid_range = exp_term * (1j * k_abs * delta_r_abs) * common_term / delta_r_abs**5
    short_range = -exp_term * common_term / delta_r_abs**5
    
    return long_range + mid_range + short_range

@jit(nopython=True, parallel=True)
def sum_direct_dipole_field(dipole_moment, position, box, k, neighbor_list):
    """Parallelized sum of dipole fields"""
    field = np.zeros((position.shape[0], 3), dtype=np.complex128)
    for i in prange(position.shape[0]):
        for j in neighbor_list[i]:
            field[i] += direct_dipole_field(
                dipole_moment[j[0]], 
                position[j[0]] + np.array([j[1], j[2], 0]), 
                position[i], 
                box, 
                k
            )
    return field

def neighbor_list(position, box):
    area_9_box = box[0]*box[1]*9
    radius_9_box = np.sqrt(area_9_box/np.pi)
    neighbor_list = []
    # loop over all particles and their replicates
    for i in range(position.shape[0]):
        neighbor_i = []
        for j in range(position.shape[0]):
            if i != j:
                for x_shift in [-box[0],0,box[0]]:
                    for y_shift in [-box[1],0,box[1]]:
                        if np.linalg.norm(position[i]-(position[j]+np.array([x_shift,y_shift,0]))) < radius_9_box:
                            neighbor_i.append([j,x_shift,y_shift])
            else:
                for x_shift in [-box[0],0,box[0]]:
                    for y_shift in [-box[1],0,box[1]]:
                        if x_shift == 0 and y_shift == 0:
                            continue
                        if np.linalg.norm(position[i]-(position[j]+np.array([x_shift,y_shift,0]))) < radius_9_box:
                            neighbor_i.append([j,x_shift,y_shift])
        neighbor_list.append(neighbor_i)
    return neighbor_list
    
def __main__():
    position = np.array([[0,0,0],[0,0,1],[0,0,2]])
    box = np.array([1,1,0])
    k = np.array([0,0,1])
    alpha = 1
    dipole_moment = Dipole_model(position, alpha, box, k)

    print(dipole_moment)



if __name__ == "__main__":
    __main__()