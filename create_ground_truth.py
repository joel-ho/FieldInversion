from constants import *

if __name__ == '__main__':
  
  # Create grid
  grid_line = np.linspace(0, 1, int(1/DELTA_H)+1)
  [x_grid, y_grid] = np.meshgrid(grid_line, grid_line)
  
  # Initialize T
  T_grid = T_BOUNDARY*np.ones(x_grid.shape)
  
  # Compute source term
  q_grid = compute_q(x_grid, y_grid)
  
  # Initialize Jacobian matrix and residual vector
  dR_dT_mat = np.zeros((x_grid.size, x_grid.size))
  R_vec = np.zeros((x_grid.size, 1))
  delta_T = np.zeros((x_grid.size, 1))
  
  # Convenience functions
  def get_T(i, j):
    if i<0 or j<0:
      return T_BOUNDARY
    else:
      try:
        return T_grid[i, j]
      except:
        return T_BOUNDARY
  
  # Store convergence
  n_vec = np.arange(0, N_ITER_MAX)
  abs_residual = np.zeros(n_vec.shape)
  
  # Newton iteration
  for iter in range(N_ITER_MAX):
  
    # Fill Jacobian matrix and residual vector
    for i in range(x_grid.shape[0]):
      for j in range(x_grid.shape[1]):
        
        input_vec = np.array([
            get_T(i, j),
            get_T(i-1, j),
            get_T(i+1, j),
            get_T(i, j-1),
            get_T(i, j+1),
            q_grid[i, j],
            BETA_1_TRUTH,
            BETA_2_TRUTH
          ])
        
        # Fill residual vector
        R_vec[ravel(i, j)] = compute_R_int(input_vec)
        
        # Fill Jacobian matrix
        grad_R = compute_R_int_grad(input_vec)
        dR_dT_mat[ravel(i, j), ravel(i, j)] = grad_R[0]
        if i > 0:
          dR_dT_mat[ravel(i, j), ravel(i-1, j)] = grad_R[1]
        if i < x_grid.shape[0]-1:
          dR_dT_mat[ravel(i, j), ravel(i+1, j)] = grad_R[2]
        if j > 0:
          dR_dT_mat[ravel(i, j), ravel(i, j-1)] = grad_R[3]
        if j < x_grid.shape[1]-1:
          dR_dT_mat[ravel(i, j), ravel(i, j+1)] = grad_R[4]
    
        # for j
      # for i
    
    delta_T = np.linalg.solve(dR_dT_mat, -R_vec)
    T_grid += OMEGA*np.reshape(delta_T, T_grid.shape)
    
    abs_residual[iter] = np.amax(np.abs(delta_T))
    # for iter
  
  # Compute k
  k_grid = compute_k([BETA_1_TRUTH, BETA_2_TRUTH, T_grid])
  
  # Save results
  with open('ground_truth.p', 'wb') as f:
    pickle.dump([x_grid, y_grid, T_grid, k_grid, q_grid], f)
  
  # Plot results
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(x_grid, y_grid, q_grid)
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(x_grid, y_grid, T_grid)
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(x_grid, y_grid, k_grid)
  
  plt.figure()
  plt.plot(n_vec, abs_residual)
  plt.grid(True)
  
  plt.show()