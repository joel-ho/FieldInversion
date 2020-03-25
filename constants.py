import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import autograd.numpy as np
from autograd import grad
import pickle

'''
  Constants
'''
DELTA_H = 0.1
T_BOUNDARY = 200
OMEGA = 0.8
N_ITER_MAX = 10
ABS_RES_TOL = 0.01

N_PROBES = 11*11
BETA_1_TRUTH = 13221
BETA_2_TRUTH = 27
N_ITER_ADJOINT = 20
OMEGA_1_ADJOINT = 10
OMEGA_2_ADJOINT = 0.0001
ADJOINT_ABS_RES_TOL = 3

OMEGA_K_ADJOINT = 0.005

'''
  Functions for solving Poisson equation with equation for k
'''
def compute_k(input_vec): # input_vec = [beta_1, beta_2, T]
  return input_vec[0]/input_vec[2] + input_vec[1]

def compute_q(x, y):
  # q = np.zeros(x.shape)
  # m = np.sqrt((x-0.5)**2+(y-0.5)**2) < 0.21
  # q[m] = 0.1e6
  q = 0.5e6*(1-(x-0.5)**2)*(1-(y-0.5)**2)
  return q

compute_k_grad = grad(compute_k)

def compute_R_int(input_vec): # input_vec = [T_i,j    T_i-1,j    T_i+1,j    T_i,j-1    T_i,j+1    q    beta_1    beta_2]
  
  R_int_term_1 = \
    compute_k(np.array([
        input_vec[6], 
        input_vec[7], 
        input_vec[0]
    ]))*\
    (
      (input_vec[2] - 2*input_vec[0] + input_vec[1])/(DELTA_H**2) + 
      (input_vec[4] - 2*input_vec[0] + input_vec[3])/(DELTA_H**2)
    )
  
  k_grad = compute_k_grad(np.array([
    input_vec[6], 
    input_vec[7], 
    input_vec[0]
  ]))
  
  R_int_term_2 = k_grad[2]*\
  (
    ((input_vec[2] - input_vec[1])/(2*DELTA_H))**2 +
    ((input_vec[4] - input_vec[3])/(2*DELTA_H))**2
  )
  
  R_int_term_3 = input_vec[5]
  
  return R_int_term_1 + R_int_term_2 + R_int_term_3
  
compute_R_int_grad = grad(compute_R_int)

ravel = lambda i, j: np.ravel_multi_index((i, j), [int(1/DELTA_H)+1, int(1/DELTA_H)+1])
unravel = lambda n: np.unravel_index(n, [int(1/DELTA_H)+1, int(1/DELTA_H)+1])


'''
  Functions for discrete k
'''
def compute_R_int_discrete_k(input_vec): # input_vec = [T_i,j    T_i-1,j    T_i+1,j    T_i,j-1    T_i,j+1    q    k_i,j    k_i-1,j    k_i+1,j    k_i,j-1    k_i,j+1]
  
  R_int_term_1 = input_vec[6]*\
    (
      (input_vec[2] - 2*input_vec[0] + input_vec[1])/(DELTA_H**2) + 
      (input_vec[4] - 2*input_vec[0] + input_vec[3])/(DELTA_H**2)
    )
  
  R_int_term_2 = \
    (
      (input_vec[8] - input_vec[7])/(2*DELTA_H)*\
      (input_vec[2] - input_vec[1])/(2*DELTA_H)
      
    )\
    +\
    (
      (input_vec[10] - input_vec[9])/(2*DELTA_H)*\
      (input_vec[4] - input_vec[3])/(2*DELTA_H)
    )
  
  R_int_term_3 = input_vec[5]
  
  return R_int_term_1 + R_int_term_2 + R_int_term_3

compute_R_int_discrete_k_grad = grad(compute_R_int_discrete_k)

def update_k_grid_ghost_points(k_grid):
  k_grid[0, :] = 2*k_grid[1, :] - k_grid[2, :]
  k_grid[-1, :] = 2*k_grid[-2, :] - k_grid[-3, :]
  k_grid[:, 0] = 2*k_grid[:, 1] - k_grid[:, 2]
  k_grid[:, -1] = 2*k_grid[:, -2] - k_grid[:, -3]
  return k_grid
  
ravel_k = lambda i, j: np.ravel_multi_index((i, j), [int(1/DELTA_H)+1+2, int(1/DELTA_H)+1+2])