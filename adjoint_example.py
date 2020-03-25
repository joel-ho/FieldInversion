from constants import *

if __name__ == '__main__':
  
  # Load ground truths
  with open('ground_truth.p', 'rb') as f:
    x_grid, y_grid, T_grid_truth, k_grid_truth, q_grid = pickle.load(f)
    
  # Extract temperatures in random locations
  probe_mask = np.random.choice(x_grid.size, N_PROBES, replace=False)
  probe_mask_grid  = unravel(probe_mask)
  
  # Set up cost function
  def compute_rms(T_grid):
    return np.sqrt(np.sum((
      T_grid[probe_mask_grid] - T_grid_truth[probe_mask_grid]
    )**2)/len(probe_mask))
  
  # Initialize conduction parameters
  beta_1 = 10000
  beta_2 = 20
  
  # Initialize variables
  T_grid = T_BOUNDARY*np.ones(x_grid.shape)
  dR_dT_mat = np.zeros((x_grid.size, x_grid.size))
  R_vec = np.zeros((x_grid.size, ))
  delta_T = np.zeros((x_grid.size, ))
  g_vec = np.zeros((x_grid.size, ))
  v_vec = np.zeros((x_grid.size, ))
  f_mat = np.zeros((x_grid.size, 2))
  dJ_dbeta_vec = np.zeros((2, ))
  
  # Convenience functions
  def get_T(i, j):
    if i<0 or j<0:
      return T_BOUNDARY
    else:
      try:
        return T_grid[i, j]
      except:
        return T_BOUNDARY
  
  # Adjoint optimization iteration
  n_adj_vec = np.arange(0, N_ITER_ADJOINT)
  adj_residual = np.nan*np.ones(n_adj_vec.shape)
  for iter_adjoint in range(N_ITER_ADJOINT):
  
    print('\nAdjoint iter %d, beta_1_current: %.4f, beta_2_current: %.4f'% ((iter_adjoint+1), beta_1, beta_2))
  
    # Newton iteration
    n_vec = np.arange(0, N_ITER_MAX)
    abs_residual = np.zeros(n_vec.shape)
    for iter in range(N_ITER_MAX): # Additional iteration to load up matrix for adjoint
    
      # Fill Jacobian matrix and residual vector
      for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
          
          input_vec = np.asarray([
              get_T(i, j),
              get_T(i-1, j),
              get_T(i+1, j),
              get_T(i, j-1),
              get_T(i, j+1),
              q_grid[i, j],
              beta_1,
              beta_2
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
      T_grid += OMEGA*np.reshape(np.squeeze(delta_T), T_grid.shape)
      
      abs_residual[iter] = np.amax(np.abs(delta_T))
      print('\tSolver iter %d: abs_residual: %.2E'% ((iter+1), abs_residual[iter]))
      if abs_residual[iter] < ABS_RES_TOL:
        break
      
      # for iter
    
    # Compute g_vec: dJ_dT
    for i_probe in probe_mask:
      g_vec[i_probe] = 2*(T_grid[unravel(i_probe)] - T_grid_truth[unravel(i_probe)])
    
    # Solve adjoint solution
    v_vec = np.linalg.solve(dR_dT_mat.T, g_vec)
    
    # Compute f_mat: dR_dT, input_vec and dR_dT_mat already loaded with latest values
    for i in range(x_grid.shape[0]):
      for j in range(x_grid.shape[1]):
        input_vec = np.asarray([
            get_T(i, j),
            get_T(i-1, j),
            get_T(i+1, j),
            get_T(i, j-1),
            get_T(i, j+1),
            q_grid[i, j],
            beta_1,
            beta_2
          ])
        f_mat[ravel(i, j), :] = -compute_R_int_grad(input_vec)[-2:]
    
    # Solve for sensitivity
    for i_beta in range(f_mat.shape[1]):
      dJ_dbeta_vec[i_beta] = np.sum(v_vec*f_mat[:, i_beta])
      
    # Compute RMS error
    rms_err_current = compute_rms(T_grid)
      
    # Update conduction parameters
    beta_1 -= OMEGA_1_ADJOINT*dJ_dbeta_vec[0]
    beta_2 -= OMEGA_2_ADJOINT*dJ_dbeta_vec[1]
    
    print('\tAdjoint iteration %d complete. rms_err_current: %.2E, beta_1_next: %.4f, beta_2_next: %.4f'% (
      (iter_adjoint+1), rms_err_current, beta_1, beta_2))
    
    adj_residual[iter_adjoint] = rms_err_current
    if rms_err_current < ADJOINT_ABS_RES_TOL:
      break
    
    # for iter_adjoint
    
  # Plot results
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x_grid[probe_mask_grid], y_grid[probe_mask_grid], T_grid_truth[probe_mask_grid], c='r', label='Truth')
  ax.plot_surface(x_grid, y_grid, T_grid, alpha=0.7, label='Adjoint')
  plt.title('Surface plot of temperature')
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(x_grid, y_grid, compute_k((beta_1, beta_2, T_grid)))
  plt.title('Surface plot of conductivity')
  
  fig = plt.figure()
  T_truth = np.linspace(200, 1200, 20)
  plt.plot(T_truth, compute_k(np.array((BETA_1_TRUTH, BETA_2_TRUTH, T_truth))),label='Truth')
  plt.scatter(T_truth, compute_k(np.array((beta_1, beta_2, T_truth))), label='Adjoint')
  plt.xlabel('T')
  plt.ylabel('k')
  plt.title('Conductivity against temperature')
  plt.grid()
  plt.legend()
  
  fig = plt.figure()
  plt.plot(n_adj_vec, adj_residual)
  plt.xlabel('Adjoint iteration')
  plt.ylabel('Adjoint objective function value')
  plt.title('Adjoint objective function')
  plt.grid()
  
  plt.show()