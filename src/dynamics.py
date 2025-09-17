import numpy as np

def relu(x):
    if x < 0:
        return 0
    else:
        return x
    
def make_exp_rise_input(num_steps, dt, stim_onset, base_current, new_current, tau_rise):
    """
    Returns an array of input current:
    - base_current before stim_onset
    - rises exponentially to new_current after stim_onset and stays there
    """
    input_array = np.full(num_steps, base_current)
    onset_idx = int(stim_onset / dt)
    # print("new current: ", new_current)
    if onset_idx < num_steps:
        t = np.arange(num_steps - onset_idx) * dt
        rise = (new_current - base_current) * (1 - np.exp(-t / tau_rise))
        input_array[onset_idx:] = base_current + rise
    return input_array

def dynamics(W, bg, I):

    duration = 10 # seconds
    dt = 10e-4
    num_steps = int(duration/dt)

    r_l = np.zeros(num_steps)
    r_e = np.zeros(num_steps)
    r_i = np.zeros(num_steps)

    w_ll, w_le, w_li = W[0, 0], W[0, 1], - W[0, 2]
    w_el, w_ee, w_ei = W[1, 0], W[1, 1], - W[1, 2]
    w_il, w_ie, w_ii = W[2, 0], W[2, 1], - W[2, 2]

    I_l, I_e, I_i = I[0], I[1], I[2]
    bg_l, bg_e, bg_i = bg[0], bg[1], bg[2]

    time_array = np.arange(0, duration, dt)


    tau_l = 2e-3
    tau_e = 2e-3
    tau_i = 2e-3 

    stim_onset = 4

    tau_rise_l = 2e-1 * (1/I_l)
    tau_rise_e = 2e-1 * (1/I_e)
    tau_rise_i = 2e-1 * (1/I_i)

    input_l = make_exp_rise_input(num_steps, dt, stim_onset, bg_l, new_current=2*bg_l, tau_rise=tau_rise_l)
    input_e = make_exp_rise_input(num_steps, dt, stim_onset, bg_e, new_current=2*bg_e, tau_rise=tau_rise_e)
    input_i = make_exp_rise_input(num_steps, dt, stim_onset, bg_i, new_current=2*bg_i, tau_rise=tau_rise_i)

    for t in range(1, num_steps):

        dr_e = -r_e[t-1] + relu(w_el*r_l[t-1] + w_ee*r_e[t-1] - w_ei*r_i[t-1] + bg_e*input_e[t-1])
        dr_l = -r_l[t-1] + relu(w_ll*r_l[t-1] + w_le*r_e[t-1] - w_li*r_i[t-1] + bg_l*input_l[t-1])
        dr_i = -r_i[t-1] + relu(w_il*r_l[t-1] + w_ie*r_e[t-1] - w_ii*r_i[t-1] + bg_i*input_i[t-1])

        r_e[t] = r_e[t-1] + (dt/tau_e)*dr_e
        r_l[t] = r_l[t-1] + (dt/tau_l)*dr_l
        r_i[t] = r_i[t-1] + (dt/tau_i)*dr_i

    return r_l, r_e, r_i, time_array