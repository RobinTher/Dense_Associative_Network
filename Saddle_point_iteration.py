import numpy as np
from scipy.integrate import quad
from scipy.integrate import simpson
from scipy.special import erfinv

import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Setting up plotting

direct_cmap = colors.ListedColormap(["mediumaquamarine", "plum", "mediumpurple", "sandybrown"])
inverse_nishimori_cmap = colors.ListedColormap(["mediumaquamarine", "plum", "cornflowerblue", "sandybrown"])
inverse_fixed_T_s_cmap = colors.ListedColormap(["mediumaquamarine", "plum", "cornflowerblue", "mediumpurple", "sandybrown"])

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def plot_phase(phases, n_beta, n_alpha, alpha_range, T_range, p = None,
               forephases = None, model = None, fontsize = 13, T_ref = None,
               draw_neg_entropy_line = False):
    '''
    Plot a phase diagram found using the saddle-point equations.
    
    Inputs
    ------
    phases (float array):
        Array of phases as a function of T (axis 0) and alpha (axis 1).
        Each phase is represented by a different number between 0 and 1.
    n_beta (int):
        Size of the phase array along axis 0.
    n_alpha (int):
        Size of the phase array along axis 1.
    alpha_range (float array):
        Range of alpha corresponding to axis 1 of the phase array.
    T_range (float array):
        Range of T corresponding to axis 0 of the phase array.
    p (int):
        Interaction order of the student. If supplied, plot p = user_value as title.
    forephases (float array):
        Array of phases to be plotted in the foreground as black lines.
    model (str):
        Adjust the color map and the color bar to the model studied.
        For the direct model, type "direct".
        For the inverse model on the Nishimori line, type "inverse_nishimori".
        For the inverse model at fixed T_s, i.e. outside of the Nishimori line, type "inverse_fixed_T_s".
        Otherwise, the color bar defaults to viridis.
    fontsize (float):
        Fontsize of all plot labels.
    T_ref (float):
        A horizontal white line is drawn at T = T_ref. Used to indicate the Nishimori line on fixed T_s digram.
    draw_neg_entropy_line (bool):
        Whether to draw the line where the entropy of the paramagnetic phase becomes negative.
    
    Outputs
    -------
    None
    '''
    
    alpha_min = 0
    alpha_max = np.max(alpha_range)
    T_min = np.around(np.min(T_range))
    T_max = np.around(np.max(T_range))
    # T_min = np.min(T_range)
    # T_max = np.max(T_range)
    
    if model == "direct":
        cmap = direct_cmap
    elif model == "inverse_nishimori":
        cmap = inverse_nishimori_cmap
    elif model == "inverse_fixed_T_s":
        cmap = inverse_fixed_T_s_cmap
    else:
        cmap = "viridis"
    
    plt.matshow(phases, vmin = 0, vmax = 1, origin = "lower", cmap = cmap)
    
    if model == "direct":
        colorbar = plt.colorbar(ticks = [0.125, 0.375, 0.625, 0.875], drawedges = True)
        colorbar.ax.set_yticklabels([r"$gR$", r"$lR$", r"$SG$", r"$P$"], fontsize = fontsize)
        plt.contour(phases, colors = "black")
        
        if p == 3:
            alpha_line = np.arange(0, n_alpha)
            
            plt.plot(alpha_line, n_beta/T_max * 0.238*np.sqrt(2 * alpha_max*alpha_line/n_alpha), color = "white",
                     linestyle = ":", linewidth = 3, zorder = 2.5)
            
            plt.plot(alpha_line, n_beta/T_max * 0.652*np.sqrt(2 * alpha_max*alpha_line/n_alpha), color = "black",
                     linestyle = ":", linewidth = 3, zorder = 2.5)
            
            plt.xlim(0, n_alpha-1)
            plt.ylim(1, n_beta)
    
    elif model == "inverse_nishimori":
        colorbar = plt.colorbar(ticks = [0.125, 0.375, 0.625, 0.875], drawedges = True)
        colorbar.ax.set_yticklabels([r"$eR$", r"$gR$", r"$lR$", r"$P$"], fontsize = fontsize)
        plt.contour(phases, colors = "black")
        
        if p == 3:
            alpha_line = np.arange(0, n_alpha)
            
            T_line_1 = n_beta/T_max * 0.652*np.sqrt(2 * alpha_max*alpha_line/n_alpha)
            T_line_2 = n_beta/T_max * 0.682*np.sqrt(2 * alpha_max*alpha_line/n_alpha)
            
            keep_T = T_line_1 > np.argmax(phases[:, 0]) + 1
            T_line_1 = T_line_1[keep_T]
            alpha_line_1 = alpha_line[keep_T]
            
            keep_T = T_line_2 > np.argmax(phases[:, 0]) + 1
            T_line_2 = T_line_2[keep_T]
            alpha_line_2 = alpha_line[keep_T]
            
            plt.plot(alpha_line_1, T_line_1, color = "white", linestyle = ":", linewidth = 3, zorder = 2.5)
            
            plt.plot(alpha_line_2, T_line_2, color = "white", linestyle = ":", linewidth = 3, zorder = 2.5)
            
            plt.xlim(0, n_alpha-1)
            plt.ylim(1, n_beta)
    
    elif model == "inverse_fixed_T_s":
        colorbar = plt.colorbar(ticks = [0.1, 0.3, 0.5, 0.7, 0.9], drawedges = True)
        colorbar.ax.set_yticklabels([r"$eR$", r"$gR$", r"$lR$", r"$SG$", r"$P$"], fontsize = fontsize)
        plt.contour(phases, colors = "black")
    else:
        plt.colorbar(ticks = [0, 1])
    
    if forephases is not None:
        for phase in forephases:
            contour = plt.contour(phase, colors = "black", linestyles = "--", linewidths = 0.95)
            
            for collection in contour.collections:
                collection.set_dashes([(0, (4, 4))])
    
    if T_ref is not None:
        plt.plot(np.full_like(alpha_range, T_ref/T_max * n_beta), color = "white", linestyle = "--", linewidth = 2)
    
    if draw_neg_entropy_line:
        alpha_line = np.arange(0, n_alpha)
        
        plt.plot(alpha_line, n_beta/T_max * np.sqrt(alpha_max*alpha_line/n_alpha / 2/np.log(2)), color = "white",
                 linestyle = "--", linewidth = 2)
        plt.xlim(0, n_alpha-1)
        plt.ylim(1, n_beta)
    
    x_labels = np.linspace(alpha_min, alpha_max, num = 5, endpoint = True)
    if np.all(x_labels == np.floor(x_labels)):
        x_labels = x_labels.astype("int32")
    
    y_labels = np.linspace(T_min, T_max, num = 5, endpoint = True)
    if np.all(y_labels == np.floor(y_labels)):
        y_labels = y_labels.astype("int32")
    
    plt.xticks(ticks = np.linspace(0, n_alpha-1, num = 5, endpoint = True),
               labels = x_labels, fontsize = fontsize)
    plt.yticks(ticks = np.linspace(0, n_beta-1, num = 5, endpoint = True),
               labels = y_labels, fontsize = fontsize)
    plt.gca().xaxis.tick_bottom()
    plt.xlabel(r"$\alpha$", fontsize = fontsize)
    plt.ylabel(r"$T$", fontsize = fontsize)
    
    if p is not None:
        plt.title(r"$p = %d$" % p, fontsize = fontsize)
    
    plt.show()

def integral(y, x):
    '''
    Definite integral of y with respect to x.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html.
    Used as a helper function.
    '''
    return simpson(y, x)

def logcosh(x):
    '''
    Numerically stable logcosh(x).
    Used as a helper function.
    '''
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)

def f_density(x, r, r_s, k, c, c_s, beta):
    '''
    Helper function of f.
    Refer to mean_field_iteration for more details about the inputs.
    '''
    return 1/np.sqrt(2*np.pi) * np.exp(-x**2/2) * (logcosh(np.sqrt(c*r) * x + c_s*r_s + beta*k)
                                                   + logcosh(np.sqrt(c*r) * x + c_s*r_s - beta*k))/2

def h_s_density(x, r, r_s, k, c, c_s, beta):
    '''
    Helper function of h_s.
    Refer to mean_field_iteration for more details about the inputs.
    '''
    return 1/np.sqrt(2*np.pi) * np.exp(-x**2/2) * (np.tanh(np.sqrt(c*r) * x + c_s*r_s + beta*k)
                                                   + np.tanh(np.sqrt(c*r) * x + c_s*r_s - beta*k))/2

def h_density(x, r, r_s, k, c, c_s, beta):
    '''
    Helper function of h.
    Refer to mean_field_iteration for more details about the inputs.
    '''
    return 1/np.sqrt(2*np.pi) * np.exp(-x**2/2) * (np.tanh(np.sqrt(c*r) * x + c_s*r_s + beta*k)**2
                                                   + np.tanh(np.sqrt(c*r) * x + c_s*r_s - beta*k)**2)/2

def s_density(x, r, r_s, k, c, c_s, beta):
    '''
    Helper function of s.
    Refer to mean_field_iteration for more details about the inputs.
    '''
    return 1/np.sqrt(2*np.pi) * np.exp(-x**2/2) * (np.tanh(np.sqrt(c*r) * x + c_s*r_s + beta*k)
                                                   + np.tanh(np.sqrt(c*r) * x - c_s*r_s + beta*k))/2

def g(q, p):
    '''
    Evaluate r_s, r and k as a function of q_s, q and m, respectively, during the saddle point iteration used to solve Eqs. (4).
    Refer to mean_field_iteration for more details about the inputs.
    '''
    return p*q**(p-1)

def f(x, q, q_s, m, p, r, r_s, k, c, c_s, beta):
    '''
    Evaluate the free entropy.
    Refer to mean_field_iteration for more details about the inputs.
    '''
    integrated_f_density = integral(f_density(x, r[..., np.newaxis], r_s[..., np.newaxis], k[..., np.newaxis],
                                              c[..., np.newaxis], c_s[..., np.newaxis], beta[..., np.newaxis]), x)
    return c_s*q_s**p - 1/2*c*q**p + beta*m**p - c_s*q_s*r_s + 1/2*c*q*r - 1/2*c*r - beta*m*k + 1/2*c + np.log(2) + integrated_f_density

def h_s(x, r, r_s, k, c, c_s, beta):
    '''
    Evaluate q_s as a function of r_s, r and k during the saddle point iteration used to solve Eqs. (4).
    Refer to mean_field_iteration for more details about the inputs.
    '''
    return integral(h_s_density(x, r[..., np.newaxis], r_s[..., np.newaxis], k[..., np.newaxis],
                                c[..., np.newaxis], c_s[..., np.newaxis], beta[..., np.newaxis]), x)

def h(x, r, r_s, k, c, c_s, beta):
    '''
    Evaluate q as a function of r_s, r and k during the saddle point iteration used to solve Eqs. (4).
    Refer to mean_field_iteration for more details about the inputs.
    '''
    return integral(h_density(x, r[..., np.newaxis], r_s[..., np.newaxis], k[..., np.newaxis],
                              c[..., np.newaxis], c_s[..., np.newaxis], beta[..., np.newaxis]), x)

def s(x, r, r_s, k, c, c_s, beta):
    '''
    Evaluate m as a function of r_s, r and k during the saddle point iteration used to solve Eqs. (4).
    Refer to mean_field_iteration for more details about the inputs.
    '''
    return integral(s_density(x, r[..., np.newaxis], r_s[..., np.newaxis], k[..., np.newaxis],
                              c[..., np.newaxis], c_s[..., np.newaxis], beta[..., np.newaxis]), x)

def mean_field_iteration(x, q_s_ref, q_ref, m_ref, r_s_ref, r_ref, k_ref,
                         n_beta, n_alpha, c_s, c, beta, p, t, init_phase, global_phase):
    '''
    Solve the saddle-point equations (Eqs. 4) via numerical iteration.
    The superscript * frequently used in the paper is replaced by _s in the code.
    
    Inputs
    ------
    x (float array):
        Range of the Gaussian variables that we integrate over in Eqs. (4).
    q_s_ref (float array):
        Values of the overlaps q_s to compare against the saddle-point solutions as a function of T (axis 0) and alpha (axis 1).
    q_ref (float array):
        Values of the overlaps q to compare against the saddle-point solutions as a function of T (axis 0) and alpha (axis 1).
    m_ref (float array):
        Values of the overlaps m to compare against the saddle-point solutions as a function of T (axis 0) and alpha (axis 1).
    r_s_ref (float array):
        Values of the overlaps r_s to compare against the saddle-point solutions as a function of T (axis 0) and alpha (axis 1).
    r_ref (float array):
        Values of the overlaps r to compare against the saddle-point solutions as a function of T (axis 0) and alpha (axis 1).
    k_ref (float array):
        Values of the overlaps k to compare against the saddle-point solutions as a function of T (axis 0) and alpha (axis 1).
    n_beta (int):
        Size of the overlap arrays along axis 0.
    n_alpha (int):
        Size of the overlap arrays along axis 1.
    c_s (float array):
        Range of beta_s*beta*alpha corresponding to the overlap arrays.
    c (float array):
        Range of beta**2*alpha corresponding to the overlap arrays.
    beta (float array):
        Range of beta corresponding to the overlap array.
    t (int):
        Number of iterations of the numerical solver.
    init_phase (str):
        Phase used as initial value for the numerical solver.
        For the ferromagnetic phase of the direct model, type "dire_ferro".
        For the ferromagnetic phase of the inverse model, type "inv_ferro".
        For the spin-glass phase, type "glass".
        For the paramagnetic phase, type "para".
    global_phase (bool):
        Whether we are looking for the globally stable phase.
        At the end of the saddle-point iteration, if global_phase == True, then q_s_ref, q_ref, m_ref, r_s_ref, r_ref and k_ref
        are updated to q_s, q, m, r_s, r and k if and only if q_s, q, m, r_s, r and k have a larger free entropy.
    
    Outputs
    -------
    q_s_ref (float array):
        Final values of the overlaps q_s as a function of T (axis 0) and alpha (axis 1).
    q_ref (float array):
        Final values of the overlaps q as a function of T (axis 0) and alpha (axis 1).
    m_ref (float array):
        Final values of the overlaps m as a function of T (axis 0) and alpha (axis 1).
    '''
    
    if init_phase == "dir_ferro":
        q_s = np.zeros((n_beta, n_alpha))
        q = np.ones((n_beta, n_alpha))
        m = np.ones((n_beta, n_alpha))
    elif init_phase == "inv_ferro":
        q_s = np.ones((n_beta, n_alpha))
        q = np.ones((n_beta, n_alpha))
        m = np.zeros((n_beta, n_alpha))
    elif init_phase == "glass":
        q_s = np.zeros((n_beta, n_alpha))
        q = np.ones((n_beta, n_alpha))
        m = np.zeros((n_beta, n_alpha))
    elif init_phase == "para":
        q_s = np.zeros((n_beta, n_alpha))
        q = np.zeros((n_beta, n_alpha))
        m = np.zeros((n_beta, n_alpha))
    else:
        raise ValueError("Value of init_phase not supported. Use 'full_ferro', 'dir_ferro', 'inv_ferro', 'glass' or 'para'.")
    
    r_s = np.zeros((n_beta, n_alpha))
    r = np.zeros((n_beta, n_alpha))
    k = np.zeros((n_beta, n_alpha))
    for t_cur in range(t):
        r_s = g(q_s, p)
        r = g(q, p)
        k = g(m, p)
        q_s = h_s(x, r, r_s, k, c, c_s, beta)
        q = h(x, r, r_s, k, c, c_s, beta)
        m = s(x, r, r_s, k, c, c_s, beta)
    
    # When we are looking for the globally stable phase, update the overlaps only if the free entropy is bigger than before
    if global_phase == True:
        criterion = f(x, q, q_s, m, p,
                      r, r_s, k, c, c_s, beta) >= f(x, q_ref, q_s_ref, m_ref, p,
                                                    r_ref, r_s_ref, k_ref, c, c_s, beta)
    # Otherwise always update
    else:
        criterion = True
    
    q_s_ref = np.where(criterion, q_s, q_s_ref)
    q_ref = np.where(criterion, q, q_ref)
    m_ref = np.where(criterion, m, m_ref)
    r_s_ref = np.where(criterion, r_s, r_s_ref)
    r_ref = np.where(criterion, r, r_ref)
    k_ref = np.where(criterion, k, k_ref)
    
    return q_s_ref, q_ref, m_ref

def disordered_phase(n_beta, n_alpha):
    '''
    Initialize q_s_ref, q_ref, m_ref, r_s_ref, r_ref and k_ref to the paramagnetic phase where they all vanish.
    
    Inputs
    ------
    n_beta (int):
        Size of the arrays along axis 0.
    n_alpha (int):
        Size of the arrays along axis 1.
    
    Outputs
    -------
    q_s_ref (float array):
        Values of the overlaps q_s to compare against the saddle-point solutions as a function of T (axis 0) and alpha (axis 1).
    q_ref (float array):
        Values of the overlaps q to compare against the saddle-point solutions as a function of T (axis 0) and alpha (axis 1).
    m_ref (float array):
        Values of the overlaps m to compare against the saddle-point solutions as a function of T (axis 0) and alpha (axis 1).
    r_s_ref (float array):
        Values of the overlaps r_s to compare against the saddle-point solutions as a function of T (axis 0) and alpha (axis 1).
    r_ref (float array):
        Values of the overlaps r to compare against the saddle-point solutions as a function of T (axis 0) and alpha (axis 1).
    k_ref (float array):
        Values of the overlaps k to compare against the saddle-point solutions as a function of T (axis 0) and alpha (axis 1).
    '''
    q_s_ref = np.zeros((n_beta, n_alpha))
    q_ref = np.zeros((n_beta, n_alpha))
    m_ref = np.zeros((n_beta, n_alpha))
    r_s_ref = np.zeros((n_beta, n_alpha))
    r_ref = np.zeros((n_beta, n_alpha))
    k_ref = np.zeros((n_beta, n_alpha))
    return q_s_ref, q_ref, m_ref, r_s_ref, r_ref, k_ref

def hyperparam_space(n_beta, n_alpha, alpha_max, T_max):
    '''
    Construct a range of alpha and T.
    
    Inputs
    ------
    n_beta (int):
        Size of the T array.
    n_alpha (int):
        Size of the alpha array.
    alpha_max (float):
        Max value of the alpha array.
    T_max (float):
        Max value of the T array.
    
    Outputs
    -------
    alpha (float array):
        Range of alpha.
    T (float array):
        Range of T.
    '''
    alpha = np.linspace(0, alpha_max, num = n_alpha, endpoint = True)
    T = np.linspace(1/n_beta * T_max, (1 + 1/n_beta) * T_max, num = n_beta, endpoint = False)
    return alpha, T