
import numpy as np
from matplotlib import pyplot as plt


def response(W, B, dI):
    """
    W : weight matrix
    I : input (dI)
    B: diagonal matrix of cellular gains
    """

    if np.linalg.det(B) == 0:
        # print("Matrix B is singular, cannot compute its inverse.")
        return np.nan, np.nan
    
    A = np.linalg.inv(B) - W

    if np.linalg.det(A) == 0:
        # print("Matrix A is singular, cannot compute its inverse.")
        return np.nan, np.nan
    R = np.linalg.inv(A)
    
    R_l = R[0] @ dI
    R_e = R[1] @ dI

    return R_l, R_e

def determinant(W, B):
    """
    Compute the determinant of the matrix A = inv(B) - W
    where B is the diagonal matrix of cellular gains
    and W is the weight matrix
    
    This determinant is the denominator of the response functions
    """
    if np.linalg.det(B) == 0:
        # print("Matrix B is singular, cannot compute its inverse.")
        return np.nan
    A = np.linalg.inv(B) - W
    det = np.linalg.det(A)
    return det

def dynamics(W, I, tau=np.ones(4), dt=0.01, duration=5):
    """
    Compute the dynamics of the system given by:
    dR/dt = -R + f(WR + I)
    where f is the linear transfer function with gain B
    """

    r = np.zeros((W.shape[0], int(duration/dt)))

    for t in range(1, r.shape[1]):
        r[:, t] = r[:, t-1] + (dt/tau) * (-r[:, t-1] + transfer_function(W @ r[:, t-1] + I))

    return r


def transfer_function(input):
    output = (1/4) * input**2
    return output

def ext_input(bg_l, bg_e, bg_p, bg_c, duration, dt, t_on, amp, tau_l, tau_e, tau_p, tau_c):
    """
    create the external input vector as a numpy array
    where at specified time, the input increases from baseline to a higher value
    in an exponentially decaying manner
    """

    n_steps = int(duration/dt)
    t_on = int(t_on/dt)
    I = np.zeros((4, int(duration/dt)))

    for t in range(n_steps):
        if t >= t_on:
            I[0, t] = bg_l + amp * (1 - np.exp(-(t - t_on)/tau_l))
            I[1, t] = bg_e + amp * (1 - np.exp(-(t - t_on)/tau_e))
            I[2, t] = bg_p + amp * (1 - np.exp(-(t - t_on)/tau_p))
            I[3, t] = bg_c + amp * (1 - np.exp(-(t - t_on)/tau_c))
        else:
            I[0, t] = bg_l
            I[1, t] = bg_e
            I[2, t] = bg_p
            I[3, t] = bg_c

    return I


def ext_input_const(bg_l=1, bg_e=1, bg_p=1, bg_c=1):
    """
    create the external input vector as a numpy array
    where the input is constant
    """

    I = np.array([bg_l, bg_e, bg_p, bg_c], dtype=float)

    return I

def ss_gain(W, r_ss, I):
    """
    Compute the steady state gain of the system
    given the weight matrix W and the steady state rates r, and the input to the system I

    assumes that the transfer function = f(input) = (input^2)/4
    """
    B_ss = np.diag((W @ r_ss + I)/2)
    return B_ss

def response_regime_metric(R_l, R_e):
    """
    Compute the response regime metric
    defined as the ratio of the change in R_e to the change in R_l
    """

    if R_l > 0 and R_e > 0: # training
        return 0
    elif R_l > 0 and R_e <= 0: # early recall
        return 1
    elif R_l <= 0 and R_e > 0: # late recall
        return 2
    else: # no response
        return 3
    
def ee_cross_diff(J, g_p, g_c, alpha, beta, I_l, I_e, I_p, I_c):

    W = np.array([[J, beta*J, -g_p*J, -g_c*J],
                   [alpha*J, J, -g_p*J, -g_c*J],
                   [J, J, -g_p*J, -g_c*J],
                   [J, J, -g_p*J, -g_c*J]])
    # simulate dynamics
    input = ext_input_const()
    r = dynamics(W, input)
    r_ss = r[:, -1]
    # infer the steady state gain
    B = ss_gain(W, r_ss, input)

    # compute the response functions
    dI = np.array([I_l, I_e, I_p, I_c])
    R_l, R_e = response(W, B, dI)
    det = determinant(W, B)

    return R_l, R_e, det

def plot_cross_ee(J, g_p, g_c, I_l, I_e, I_p, I_c, vmin, vmax, savefig=False, location=""):
    
    alpha_arr = np.linspace(0.1, 5.0, 100)
    beta_arr = np.linspace(0.1, 5.0, 100)

    R_l = np.zeros((len(alpha_arr), len(beta_arr)))
    R_e = np.zeros((len(alpha_arr), len(beta_arr)))
    determinant_grid = np.zeros((len(alpha_arr), len(beta_arr)))
    response_regime = np.zeros((len(alpha_arr), len(beta_arr)))

    for i, alpha in enumerate(alpha_arr):
        for j, beta in enumerate(beta_arr):
            R_l[i, j], R_e[i, j], determinant_grid[i, j] = ee_cross_diff(J, g_p, g_c, alpha, beta, I_l, I_e, I_p, I_c)
            response_regime[i, j] = response_regime_metric(R_l[i, j], R_e[i, j])

    fig, ax = plt.subplots(1, 4, figsize=(10, 2.5), dpi=200)

    im1 = ax[0].imshow(determinant_grid.T, extent=(alpha_arr.min(), alpha_arr.max(), beta_arr.min(), beta_arr.max()), origin='lower', aspect='auto')
    ax[0].set_title('Determinant')
    ax[0].set_xlabel(r'$\alpha$')
    ax[0].set_ylabel(r'$\beta$')
    fig.colorbar(im1, ax=ax[0])
    # add dashed white line at determinant = 0
    cx = ax[0].contour(alpha_arr, beta_arr, determinant_grid.T, levels=[0], colors='white', linestyles='dashed')
    cx.clabel(fmt='%.2f', inline=True, fontsize=8)

    im2 = ax[1].imshow(R_l.T, extent=(alpha_arr.min(), alpha_arr.max(), beta_arr.min(), beta_arr.max()), origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    ax[1].set_title('R_l')
    ax[1].set_xlabel(r'$\alpha$')
    ax[1].set_ylabel(r'$\beta$')
    fig.colorbar(im2, ax=ax[1])
    cx = ax[1].contour(alpha_arr, beta_arr, R_l.T, levels=[0], colors='white', linestyles='dashed')
    cx.clabel(fmt='%.2f', inline=True, fontsize=8)

    im3 = ax[2].imshow(R_e.T, extent=(alpha_arr.min(), alpha_arr.max(), beta_arr.min(), beta_arr.max()), origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    ax[2].set_title('R_e')
    ax[2].set_xlabel(r'$\alpha$')
    ax[2].set_ylabel(r'$\beta$')
    fig.colorbar(im3, ax=ax[2])
    cx = ax[2].contour(alpha_arr, beta_arr, R_e.T, levels=[0], colors='white', linestyles='dashed')
    cx.clabel(fmt='%.2f', inline=True, fontsize=8)

    im4 = ax[3].imshow(response_regime.T, extent=(alpha_arr.min(), alpha_arr.max(), beta_arr.min(), beta_arr.max()), origin='lower', aspect='auto')
    ax[3].set_title('Response Regime')
    ax[3].set_xlabel(r'$\alpha$')
    ax[3].set_ylabel(r'$\beta$')
    cbar = fig.colorbar(im4, ax=ax[3], ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Training', 'Early Recall', 'Late Recall', 'No Response']) 

    fig.suptitle('Excitatory cross connections')
    fig.tight_layout()

    if savefig:
        plt.savefig(location + "cross_ee.svg")
    plt.show()

    # plt.figure(figsize=(6, 5), dpi=200)
    # plt.imshow(response_regime.T, extent=(alpha_arr.min(), alpha_arr.max(), beta_arr.min(), beta_arr.max()), origin='lower', aspect='auto')
    # plt.title('Response Regime')
    # plt.xlabel(r'$\alpha$')
    # plt.ylabel(r'$\beta$')
    # plt.colorbar(ticks=[0, 1, 2, 3], labels=['Training', 'Early Recall', 'Late Recall', 'No Response'])
    # plt.show()
    
    return response_regime