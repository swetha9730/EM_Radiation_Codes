import numpy as np
from scipy.optimize import fsolve

c = 3 * 10**10 # cm/s #Speed of light
q = 1 # Charge in esu

def retarded_time_eq(t_source, t_obs, r_obs, r_o_func):
    """"
    Computes the retarted time equation equating to zero as fsolve returns the roots of the function equal to zero.
    
    t_source: source time => retarted time
    t_obs: observer/field time
    r_obs: observer/field position
    """
    r_obs = r_obs.astype(float)
    r_o_t_source = r_o_func(t_source)
    distance = np.linalg.norm(r_obs - r_o_t_source)
    return t_obs - distance / c - t_source

def v_o(t, r_o_func, delta=1e-14):
    """
    Computes the velocity v_o(t) by finite differencing r_o(t).

    t: time
    r_o_func: function that returns the position of the charge at time t
    delta: small number for finite differencing
    """
    t = np.asarray(t).item() #forces t to be a scalar
    r_plus = r_o_func(t + delta)
    r_minus = r_o_func(t - delta)

    velocity = (r_plus - r_minus) / (2 * delta)
    #print(f"r_plus: {r_plus}; r_minus: {r_minus}; velocity: {velocity}")
    return velocity

def a_o(t, r_o_func, delta=1e-14):
    """
    Computes the acceleration a_o(t) by finite differencing v_o(t).

    Parameters:
        t: time
        r_o_func: function that gives r_o(t)
        delta: small time step for finite difference

    Returns:
        numpy array: acceleration vector at time t
    """
    t = np.asarray(t).item()
    v_plus = v_o(t + delta, r_o_func, delta)
    v_minus = v_o(t - delta, r_o_func, delta)
    a = (v_plus - v_minus) / (2 * delta)
    return a


def spherical_to_cartesian(r, theta, phi):
    """
    Converts spherical coordinates to Cartesian coordinates.

    Parameters:
        r: radius
        theta: polar angle (inclination)
        phi: azimuthal angle
    Returns:
        numpy array: Cartesian coordinates (x, y, z)
    """

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def cartesian_to_spherical(vec):
    """
    Converts Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).

    Parameters:
        x, y, z: Cartesian coordinates

    Returns:
        r: radius
        theta: polar angle (0 <= theta <= pi)
        phi: azimuthal angle (0 <= phi < 2*pi)
    """

    r = np.linalg.norm(vec)
    x, y, z = vec
    # Avoid division by zero in theta calculation
    if r == 0:
        theta = 0.0
    else:
        theta = np.arccos(z / r)
    
    # Use arctan2 for full 360 degree range
    phi = np.arctan2(y, x)

    # Ensure phi is between 0 and 2pi
    if phi < 0:
        phi += 2 * np.pi

    return r, theta, phi

##Liénard–Wiechert Potentials##
def LW_potential(r_obs, t_obs, r_o_func, delta = 1e-12, theta=0,phi=0, q=q):
    """
    Computes Liénard–Wiechert potentials at a point r_obs and time t_obs.

    Parameters:

    Returns:

    """

    if not isinstance(r_obs, np.ndarray):
        #print(f"Note: r_obs is being interpreted as radial distance and θ = {theta} and φ = {phi}.")
        r_obs = spherical_to_cartesian(r_obs, theta, phi)
        
    r_obs = r_obs.astype(float)
    
    t_guess = t_obs - np.linalg.norm(r_obs) / c
    t_source = fsolve(retarded_time_eq, t_guess, args=(t_obs, r_obs, r_o_func))[0]

    r_source = r_o_func(t_source)
    v_source = v_o(t_source, r_o_func, delta)

    #print(f"t_source: {t_source}; r_source: {r_source}; v_source: {v_source}")
    R = r_obs - r_source
    R_mag = np.linalg.norm(R)
    n_hat = R / R_mag

    beta = v_source / c
    n_dot_beta = np.dot(n_hat, beta)

    tmp = (1 - n_dot_beta) * R_mag

    phi = q / tmp
    A = q * beta / tmp
    return phi, A

##E and B fields##
def compute_E_B(r_obs, t_obs, r_o_func, delta=1e-12,theta=0, phi=0, q=q):

    if not isinstance(r_obs, np.ndarray):
        #print(f"Note: r_obs is being interpreted as radial distance and θ = {theta} and φ = {phi}.")
        r_obs = spherical_to_cartesian(r_obs, theta, phi)

    r_obs = r_obs.astype(float)

    t_guess = t_obs - np.linalg.norm(r_obs) / c
    t_source = fsolve(retarded_time_eq, t_guess, args=(t_obs, r_obs, r_o_func))[0]
    
    r_source = r_o_func(t_source)
    v_source = v_o(t_source, r_o_func, delta)
    a_source = a_o(t_source, r_o_func, delta)

    #print(f"t_source: {t_source}; r_source: {r_source}; v_source: {v_source}; a_source: {a_source}")

    R = r_obs - r_source
    R_mag = np.linalg.norm(R)
    n_hat = R / R_mag

    beta = v_source / c
    beta_dot = a_source / c
    n_dot_beta = np.dot(n_hat, beta)

    tmp = 1 - n_dot_beta
    tmp1 = n_hat - beta

    e_1 = (tmp1 * (1 - np.power((np.linalg.norm(beta)), 2))) / (np.power(tmp, 3) * np.power(R_mag,2))
    e_2 = (np.cross(n_hat,np.cross(tmp1,beta_dot))) / (c * np.power(tmp,3) * R_mag)

    #print(f"e_1: {e_1}; e_2: {e_2}")
    E = q * (e_1 + e_2)

    #print(f"R: {R}; R_mag: {R_mag}; n_hat: {n_hat}; beta: {beta}; beta_dot: {beta_dot}; n_dot_beta: {n_dot_beta}")
    B = np.cross(n_hat, E)

    return E, B

def compute_E_B_fd(r_obs, t_obs, r_o_func, delta=1e-12, theta=0, phi=0, q=1):
    """
    Compute E and B fields using finite differencing methods.
    """

    if not isinstance(r_obs, np.ndarray):
        #print(f"Note: r_obs is being interpreted as radial distance and θ = {theta} and φ = {phi}.")
        r_obs = spherical_to_cartesian(r_obs, theta, phi)

    r_obs = r_obs.astype(float)

    #Computing the potentials at slightly pertubed times
    grad_phi = np.zeros(3)
    dA = np.zeros((3,2,3))

    delta = 1e-2

    for i in range(3):
        r_plus = np.copy(r_obs)
        r_minus = np.copy(r_obs)
        r_plus[i] += delta
        r_minus[i] -= delta

        phi_plus, A_plus = LW_potential(r_obs=r_plus, t_obs=t_obs, r_o_func=r_o_func, q=q)
        phi_minus, A_minus = LW_potential(r_obs=r_minus, t_obs=t_obs, r_o_func=r_o_func, q=q)

        grad_phi[i] = (phi_plus - phi_minus) / (2 * delta)
        dA[i][0] = A_plus
        dA[i][1] = A_minus

    A_t_plus = LW_potential(r_obs=r_obs, t_obs =t_obs + delta, r_o_func=r_o_func, q=q)[1]
    A_t_minus = LW_potential(r_obs=r_obs, t_obs=t_obs - delta, r_o_func=r_o_func, q=q)[1]

    dA_dt = (A_t_plus - A_t_minus) / (2 * delta)

    dAz_dy = (dA[1][0][2] - dA[1][1][2]) / (2 * delta)
    dAy_dz = (dA[2][0][1] - dA[2][1][1]) / (2 * delta)

    #print(dA)

    dAx_dz = (dA[2][0][0] - dA[2][1][0]) / (2 * delta)
    dAz_dx = (dA[0][0][2] - dA[0][1][2]) / (2 * delta) 

    dAy_dx = (dA[0][0][1] - dA[0][1][1]) / (2 * delta)
    dAx_dy = (dA[1][0][0] - dA[1][1][0]) / (2 * delta)

    curl_A = np.array([dAz_dy - dAy_dz, dAx_dz - dAz_dx, dAy_dx - dAx_dy])

    E = - grad_phi - (1 / c) * dA_dt
    B = curl_A
    #print("dA", dA)
    return E, B

##Poynting vector and time evolutions##

def Poynting_vector(r_obs, t_obs, r_o_func, delta=1e-12, theta=0, phi=0, q=q, method="exact"):

    if not isinstance(r_obs, np.ndarray):
        #print(f"Note: r_obs is being interpreted as radial distance and θ = {theta} and φ = {phi}.")
        r_obs = spherical_to_cartesian(r_obs, theta, phi)

    r_obs = r_obs.astype(float)

    if method == "exact":
        E, B = compute_E_B(r_obs, t_obs, r_o_func, delta=delta, q=q)
    elif method == "fd":
        E, B = compute_E_B_fd(r_obs, t_obs, r_o_func, delta=1e-2, q=q)
    else:
        raise ValueError("method must be 'exact' or 'fd'")

    S = c * np.cross(E, B) / (4 * np.pi)
    return S

def compute_E_B_vs_time(r_obs, times, r_o_func, delta=1e-12, theta=0, phi=0, q=q ,method="exact"):
    if not isinstance(r_obs, np.ndarray):
        #print(f"Note: r_obs is being interpreted as radial distance and θ = {theta} and φ = {phi}.")
        r_obs = spherical_to_cartesian(r_obs, theta, phi)

    r_obs = r_obs.astype(float)

    E_t = np.zeros(len(times))
    B_t = np.zeros(len(times))

    for i, t_obs in enumerate(times):
        if method == "exact":
            E, B = compute_E_B(r_obs, t_obs,r_o_func,delta=delta, q=q)
        elif method == "fd":
            E, B = compute_E_B_fd(r_obs, t_obs, r_o_func, delta=1e-2, q=q)
        else:
            raise ValueError("method must be 'exact' or 'fd'")

        E_t[i] = np.linalg.norm(E)
        B_t[i] = np.linalg.norm(B)

        #E_t[i] = E[1]
        #B_t[i] = B[1]

    return E_t, B_t

def compute_S_dot_r(time_array, r_obs, r_o_func, delta=1e-12,theta=0, phi=np.pi/2, q=q,method="exact"):
    if not isinstance(r_obs, np.ndarray):
        #print(f"Note: r_obs is being interpreted as radial distance and θ = {theta} and φ = {phi}.")
        r_obs = spherical_to_cartesian(r_obs, theta, phi)

    r_obs = r_obs.astype(float)

    S_dot_r = np.zeros(len(time_array))

    for i, t_obs in enumerate(time_array):
        S = Poynting_vector(r_obs, t_obs, r_o_func, delta=delta, q=q,method=method)
        S_dot_r[i] = np.dot(S, r_obs) / np.linalg.norm(r_obs)

    return S_dot_r

def compute_avg_dP_dOmega(r_obs, r_o_func, time_array,delta=1e-12,theta=0, phi=0, q=q, method="exact"):
    """
    Computes the angular power distribution at a point r_obs.

    Parameters:
        r_obs: observer position
        r_o_func: function that gives r_o(t)
        theta: polar angle
        phi: azimuthal angle
        q: charge
    Returns:
        numpy array: angular power distribution
    """
    if not isinstance(r_obs, np.ndarray):
        #print(f"Note: r_obs is being interpreted as radial distance and θ = {theta} and φ = {phi}.")
        r_obs = spherical_to_cartesian(r_obs, theta, phi)

    r_obs = r_obs.astype(float)

    #if time_array == 0:
    #    time_array = np.linspace(0, 0.5, 100)

    S_dot_r = compute_S_dot_r(time_array, r_obs, r_o_func, delta=delta, theta=theta, phi=phi, method=method)
    dP_dOmega = np.mean(S_dot_r) * np.power(np.linalg.norm(r_obs), 2)

    return dP_dOmega

def compute_dP_dw(time_array, r_obs, r_o_func, delta=1e-12, theta=0, phi=np.pi/2, q=q, method="exact"):
    """
    Computes the power distribution over frequency.

    Parameters:
        r_obs: observer position
        r_o_func: function that gives r_o(t)
        theta: polar angle
        phi: azimuthal angle
        q: charge
    Returns:
        numpy array: power distribution over frequency
    """

    S_dot_r = compute_S_dot_r(time_array=time_array, r_obs=r_obs, r_o_func=r_o_func, delta=delta, theta=theta, phi=phi, q=q, method=method) 
    dt = time_array[1] - time_array[0]
    S_dot_r_fft = np.fft.fft(S_dot_r)
    freq = np.fft.fftfreq(len(S_dot_r), dt)

    dP_dw_dOmega = 4 * np.pi * np.abs(S_dot_r_fft) * (np.linalg.norm(r_obs))**2
    return freq, dP_dw_dOmega






