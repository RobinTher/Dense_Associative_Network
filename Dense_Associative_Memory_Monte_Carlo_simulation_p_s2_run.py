import numpy as np
import jax
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

from Dense_Associative_Memory_Monte_Carlo_simulation import *

def p_s2_run(p, seed, N, L, t):
    '''
    Run Monte-Carlo simulations for p_s = 2 and a predefined range of T and alpha to reproduce Fig. (7).
    The results are saved in .npy files.
    
    Inputs
    ------
    p (int):
        Interaction order of the student network.
    seed (int):
        User-defined seed for random number generation.
    N (int):
        Number of entries in the patterns memorized by the student network and generated by the teacher network.
    L (int):
        Number of patterns sampled from the student network.
    t (int):
        Number of Monte-Carlo steps for the teacher and student networks.
    
    Outputs
    -------
    None
    '''
    p_s = 2
    
    n_beta = 20
    n_alpha = 3
    
    T_range = np.linspace(0.4, 4.4, num = n_beta, endpoint = False)
    alpha_range = np.array([0.5, 1, 1.5])
    
    M = int(np.max(alpha_range) * N**(p-1)/np.math.factorial(p))
    
    key = jax.random.PRNGKey(seed)
    key_teacher, key_student = jax.random.split(key)
    
    mean_xi_overlaps = np.zeros((n_beta, n_alpha))
    std_xi_overlaps = np.zeros((n_beta, n_alpha))
    
    teacher_init_overlap = 0
    student_init_overlap = 1
    
    teacher_batch_size = int(np.sqrt(N))
    student_batch_size = int(np.sqrt(N))
    
    teacher = Model("gardner", p_s, N, 1, M, teacher_batch_size, key_teacher)
    student = Model("gardner", p, N, M, L, student_batch_size, key_student)
        
    xi_s_spins = None
    for i, T in enumerate(T_range):
        
        T_s = np.sqrt(N * T / 2)
        if T_s == 0:
            beta_s = np.inf
        else:
            beta_s = 1/T_s
        
        xi_s_spins, sigma_spins = teacher.init_spins(teacher_init_overlap, ori_spins = xi_s_spins)
        
        sigma_spins = teacher.generate_spins(t, beta_s, xi_s_spins, sigma_spins)
        # print("Teacher done")
        
        for j, alpha in enumerate(alpha_range):
            if T == 0:
                beta = np.inf
            else:
                beta = 1/T
            
            M = int(alpha * N**(p-1)/np.math.factorial(p))
            
            xi_s_spins, xi_spins = student.init_spins(student_init_overlap, ori_spins = xi_s_spins)
            
            xi_spins = student.generate_spins(t, beta, sigma_spins.T[: M], xi_spins)
            # print("Student done")
            xi_overlaps = (xi_s_spins @ xi_spins)/N
            mean_xi_overlaps[i, j] = np.mean(xi_overlaps)
            std_xi_overlaps[i, j] = np.std(xi_overlaps)
    
    with open("./Data/overlaps/mean_xi_overlap_p_s=2_p=%d.npy" % p, "wb") as file:
        np.save(file, mean_xi_overlaps)

    with open("./Data/overlaps/std_xi_overlap_p_s=2_p=%d.npy" % p, "wb") as file:
        np.save(file, std_xi_overlaps)