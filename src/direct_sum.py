import numpy as np
import cProfile
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
#Dipole model
#epsilon_medium = 1
#position: N_particles,3, particle positions
#alpha: N_particles, particle polarizability
#box: 3,3, box size
#k: 3, wavevector of the incident light , firstly I assume it is in the z direction k= [0,0,k_z]
#E0: 2,3, electric field of the incident light at two different directions perpendicular to wavevector
def Direct_sum_Dipole_model(position, alpha, k):
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
        print(dipole_moment_guess)
        dipole_moment_solution = Solve_Dipole(position, dipole_moment_guess, k, E, alpha)
        dipole_moment[:,:,i] = dipole_moment_solution
    return dipole_moment
def Solve_Dipole(position, dipole_moment_guess, k, E, alpha):
    #Solve the dipole moment using the dipole moment guess
    dipole_moment_solution = dipole_moment_guess
    n = position.shape[0] * 3  # Multiply by 3 for x,y,z components
    A_operator = LinearOperator((n,n), matvec=make_matvec(alpha, position, k, E))
    
    # Reshape input for GMRES
    flat_guess = dipole_moment_guess.reshape(-1)
    solution = gmres(A_operator, 0, rtol=1e-6, x0=flat_guess)[0]
    
    # Reshape solution back to original shape
    dipole_moment_solution = solution.reshape(-1, 3)
    return dipole_moment_solution
def make_matvec(alpha,position, k, E):
    def matvec(x):
        # Reshape input vector to (N,3) array
        x_reshaped = x.reshape(-1, 3)
        # Calculate field
        result = alpha * Total_field(x_reshaped, position, k, E) - x_reshaped
        # Flatten result for GMRES
        return result.reshape(-1)
    return matvec
def Total_field(dipole_moment, position, k, E):
    field = Total_dipole_field(dipole_moment, position, k)
    field += E
    return field
def Total_dipole_field(dipole_moment, position, k):
    field = sum_direct_dipole_field(dipole_moment, position, k)
    return field

def direct_dipole_field(dipole_moment, dipole_position, position, k):
    delta_r = position-dipole_position
    delta_r_abs = np.linalg.norm(delta_r)
    epsilon_medium = 1
    k_abs = np.linalg.norm(k)
    long_range_term = - k_abs**2*np.exp(1j*k_abs*delta_r)*(delta_r*(np.dot(dipole_moment,delta_r))-dipole_moment*delta_r_abs**2)/(delta_r_abs**3)
    mid_range_term = np.exp(1j*k_abs*delta_r_abs)*(1j*k_abs*delta_r_abs)*(delta_r_abs**2*dipole_moment-3*delta_r*(np.dot(delta_r,dipole_moment)))/(delta_r_abs**5)
    short_range_term = - np.exp(1j*k_abs*delta_r_abs)*(delta_r_abs**2*dipole_moment-3*delta_r*(np.dot(delta_r,dipole_moment)))/(delta_r_abs**5)
    return long_range_term+mid_range_term+short_range_term

def sum_direct_dipole_field(dipole_moment, position, k):
    # Initialize field as complex array instead of real
    field = np.zeros((position.shape[0],3), dtype=complex) 
    for i in range(position.shape[0]):
        for j in range(position.shape[0]):
            if i != j:
                field[i] += direct_dipole_field(dipole_moment[j], position[j], position[i], k)
    return field

     
def __main__():
    position = np.array([[0,0,0],[0,0,1],[0,0,2]])
    k = np.array([0,0,0.01])
    alpha = 1
    dipole_moment = Direct_sum_Dipole_model(position, alpha, k)
    print(dipole_moment)

if __name__ == "__main__":
    __main__()