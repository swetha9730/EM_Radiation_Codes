import numpy as np
import matplotlib.pyplot as plt
import EM_radiation_code as em
from scipy.optimize import fsolve
import fractions

# Constants
l = 1 #amplitude of oscillation
c = 3 * 10**10 #speed of light in cm/s
w = 0.01 * c / l #angular frequency
q = 1 #charge


#data to test
r_test = np.array([1.0, 0.0, 0.0])
t_test = 0.0
radius_test = 1.0

# Define the position function r_o(t)
def r_o(t,w=w, l=l):
    """
    returns the position of the charge at time t performing oscillatory motion
    along z-axis.

    Parameters:
    t -> time
    w -> angular frequency
    l -> amplitude of oscillation
    """
    t = np.asarray(t).item() #forces t to be a scalar
    return np.array([0, 0, l * np.cos(w * t)])

#testing the retarded time function
t_source_test = fsolve(em.retarded_time_eq, 0.0, args=(np.sqrt(2) / c, r_test, r_o))
print("retarded time: ", t_source_test)
print("observer time: ", np.sqrt(2)/c)

#Computing retarded time for 2D grid of field points
x = np.linspace(-100, 100, 200)
y = np.linspace(-100, 100, 200)

def plot_retarded_time_xy_plane(t_obs = t_test, x=x, y=y, r_o_func=r_o):
    """
    Plots the retarded time in the xy plane for a given t_test.
    """
    X, Y = np.meshgrid(x, y)
    t_source = np.zeros_like(X)
    t_source_distance = np.zeros_like(X)

    for i in range(len(x)):
        for j in range(len(y)):
            r_obs = np.array([X[i, j], Y[i, j], 0])
            t_source_guess = t_obs - np.linalg.norm(r_obs) / c
            t_source[i, j] = fsolve(em.retarded_time_eq, t_source_guess, args=(t_obs, r_obs, r_o_func))[0]
            t_source_distance[i, j] = t_source[i, j] + np.linalg.norm(r_obs) / c

    #Plotting the 2D color map of retarded time
    plt.figure(figsize=(6, 5))

    extent = [X.min(), 2*X.max(), Y.min(), 2*Y.max()] 
    plt.imshow(t_source_distance, extent=extent, origin='lower', aspect='auto')
    plt.colorbar(label=r"$t_r + \frac{|\mathbf{x}|}{c}$")

    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$y$", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(r"$t_r + \frac{|\mathbf{x}|}{c}$ in the $xy$-Plane", fontsize=14)
    plt.tight_layout()
    plt.show()

#plot_retarded_time_xy_plane(t_test, x, y, r_o_func=r_o)

def plot_retarded_time_xz_plane(t_obs = t_test, x=x, z=y, r_o_func=r_o):
    """
    Plots the retarded time in the xz plane for a given t_test.
    """
    X, Z = np.meshgrid(x, z)
    t_source = np.zeros_like(X)
    t_source_distance = np.zeros_like(X)

    for i in range(len(x)):
        for j in range(len(z)):
            r_obs = np.array([X[i, j], 0, Z[i, j]])
            t_source_guess = t_obs - np.linalg.norm(r_obs) / c
            t_source[i, j] = fsolve(em.retarded_time_eq, t_source_guess, args=(t_obs, r_obs, r_o_func))[0]
            t_source_distance[i, j] = t_source[i, j] + np.linalg.norm(r_obs) / c

    #Plotting the 2D color map of retarded time
    plt.figure(figsize=(6, 5))
    #plt.pcolormesh(X, Z, t_source_distance, shading='auto') 

    extent = [X.min(), X.max(), Z.min(), Z.max()] 
    plt.imshow(t_source_distance, extent=extent, origin='lower', aspect='auto')

    plt.colorbar(label=r"$t_r + \frac{|\mathbf{x}|}{c}$")
    plt.xlabel("$x$", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("$z$", fontsize=12)
    plt.title(r"$t_r + \frac{|\mathbf{x}|}{c}$ in the $xz$-Plane", fontsize=14)
    plt.tight_layout()
    plt.show()

#plot_retarded_time_xz_plane(t_test, x, y,r_o_func=r_o)

#testing LW potetial function
phi_test, A_test = em.LW_potential(r_obs=r_test, t_obs=t_test,r_o_func=r_o)
print(f"\nphi: {phi_test}; A: {A_test}" )

#testing E, B fields
E, B = em.compute_E_B(r_obs=r_test, t_obs=t_test, r_o_func=r_o, theta=0)
print(f"\nElectric Field: {E} \nMagnetic Field: {B}")

E_fd, B_fd = em.compute_E_B_fd(r_obs=r_test, t_obs=t_test, r_o_func=r_o, delta=1e-12)
print(f"\nElectric Field (FD): {E_fd} \nMagnetic Field (FD): {B_fd}")

#plotting E and B fields vs time
def plot_E_B_vs_time(r_obs, times, r_o_func=r_o, delta=1e-14, method='exact'):
    """
    Plots |E(t)| and |B(t)| vs time at a fixed observation point.

    Parameters:
        r_obs: fixed observation point
        times: array of times
        r_o: function that returns the position of the charge at time t
        delta: small number for finite difference method
        method: "exact" or "fd"
    """
    E_t, B_t = em.compute_E_B_vs_time(r_obs=r_obs, times=times, r_o_func=r_o_func, delta=delta, method=method)
    radius = np.linalg.norm(r_obs)
    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    
    # Plot E(t)
    ax[0].plot(times, E_t, label=r"$|E(t)|$", color='blue', linewidth=2)
    ax[0].set_xlabel("Time (s)", fontsize=10)
    ax[0].set_ylabel(r"$|\mathbf{E}(t)|$ (G)", fontsize=10)
    ax[0].set_title(f"Electric Field Magnitude at r = {radius}cm", fontsize=12)
   # ax[0].legend()
    ax[0].grid(True)

    # Plot B(t)
    ax[1].plot(times, B_t, label=r"$|B(t)|$", color='red', linewidth=2)
    ax[1].set_xlabel("Time (s)", fontsize=10)
    ax[1].set_ylabel(r"$|\mathbf{B}(t)|$ (G)", fontsize=10)
    ax[1].set_title(f"Magnetic Field Magnitude at r = {radius}cm", fontsize=12)
    #ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

# Define a range of times
T =  2* np.pi / w  # Oscillation period
times_test = np.linspace(0, T, 100)  # 2 periods

#plot_E_B_vs_time(r_obs=np.array([10,0,0]), times=times_test, r_o_func=r_o, delta=1e-12, method='fd')

#Plotting Poynting vector dot r_hat
def plot_S_dot_r(r_obs, times, r_o, delta=1e-12, theta=0, phi=0, method='exact'):
    """
    Plots S(t)·r_hat vs time at a fixed observation point.

    Parameters:
        r_obs: fixed observation point
        times: array of times
        r_o: function that returns the position of the charge at time t
        delta: small number for finite difference method
        theta: polar angle
        phi: azimuthal angle
        method: "exact" or "fd"
    """

    if not isinstance(r_obs, np.ndarray):
        print(f"Note: r_obs is being interpreted as radial distance and θ = {theta} and φ = {phi}.")
        radius, theta_, phi_ = r_obs, theta, phi
        
    elif isinstance(r_obs, np.ndarray):
        #r_obs = em.spherical_to_cartesian(radius, theta_, phi_)
        radius, theta_, phi_ = em.cartesian_to_spherical(r_obs)

    S_dot_r = em.compute_S_dot_r(time_array=times, r_obs=r_obs, r_o_func=r_o, delta=delta, theta=theta, phi=phi, method=method)

    plt.figure(figsize=(8, 4))
    plt.plot(times, S_dot_r)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel(r"$\mathbf{S} \cdot \hat{r}$ (erg/cm$^2$/s)", fontsize=12)
    plt.title(f"Poynting Vector at r = {radius} cm, θ = {theta_}, φ = {phi_}", fontsize=14)
    #plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#plot_S_dot_r(r_obs=100, times=times_test, r_o=r_o, delta=1e-12, theta=np.pi/4, phi=0, method='exact')

def theta_to_latex(theta):
    """
    Converts a theta value in radians to a LaTeX-formatted string like π/4 or 3π/2.
    """
    frac = fractions.Fraction(theta / np.pi).limit_denominator()
    numerator = frac.numerator
    denominator = frac.denominator

    if numerator == 0:
        return "0"
    elif numerator == denominator:
        return r"\pi"
    elif numerator == 1:
        return rf"\frac{{\pi}}{{{denominator}}}"
    else:
        return rf"\frac{{{numerator}\pi}}{{{denominator}}}"

def plot_S_dot_r_vs_theta(radial_distance, times, r_o_func, theta_vals=np.array([0, np.pi/4, np.pi/2]), delta=1e-12, phi=0, method='exact'):
    """
    Plots S(t)·r_hat vs time at a fixed observation point for different θ values.

    Parameters:
        radial_distance: fixed radial distance
        times: array of times
        r_o: function that returns the position of the charge at time t
        theta_vals: array of polar angles
        delta: small number for finite difference method
        phi: azimuthal angle
        method: "exact" or "fd"
    """
    n = len(theta_vals)
    fig, axs = plt.subplots(n, 1, figsize=(8, 7))

    if n == 1:
        axs = [axs]  # Ensure it's always iterable

    for i, theta in enumerate(theta_vals):
        S_dot_r = em.compute_S_dot_r(time_array=times, r_obs=radial_distance, r_o_func=r_o_func, delta=delta, theta=theta, phi=phi, method=method)
        axs[i].plot(times, S_dot_r, linewidth=2.5)
        axs[i].set_ylabel(r"$\vec{S} \cdot \hat{r}$ (erg/cm$^2$/s)", fontsize=12)
        axs[i].set_title(fr"Poynting Flux at $\theta = {theta_to_latex(theta)}$", fontsize=16)
        axs[i].grid(True)

    axs[-1].set_xlabel("Time (s)", fontsize=12)
    #fig.suptitle(fr"Poynting Flux vs Time at Radius $r = {radial_distance}$ cm", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

#plot_S_dot_r_vs_theta(radial_distance=100, times=times_test, r_o_func=r_o, theta_vals=np.array([0, np.pi/4, np.pi/2]), delta=1e-12, phi=0, method='exact')

def plot_angular_power_distribution(r_obs, r_o, times, n_theta=100, delta=1e-12, phi=0, q=q, method="exact"):
    """
    Plots the angular power distribution of the Poynting vector.
    r: radius of the sphere
    n_theta: number of theta points
    phi: azimuthal angle
    """
    theta_vals = np.linspace(0, np.pi, n_theta)
    dP_domega = np.zeros(n_theta)
    r = np.linalg.norm(r_obs)

    for i, theta in enumerate(theta_vals):
        S_dot_r = em.compute_S_dot_r(time_array=times, r_obs=r, r_o_func=r_o, theta=theta, phi=phi, delta=delta, q=q, method=method)
        dP_domega[i] = np.mean(S_dot_r) * np.power(r, 2)
        #em.angular_power_distribution
    
    plt.figure(figsize=(6, 4))
    plt.plot(theta_vals, dP_domega, linewidth=2.5)
    plt.xlabel(r"$\theta$ (rad)", fontsize=12)
    plt.ylabel(r"$\frac{dP}{d\Omega}$ (erg/s/sr)", fontsize=12)
    plt.title("Angular Power Distribution at r = {:.0f} cm".format(r), fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#plot_angular_power_distribution(r_obs=10, r_o=r_o, times=times_test, n_theta=100, delta=1e-12, phi=0, q=q, method="exact")

def plot_angular_power_distribution_compare(r_obs, r_o, times, n_theta=100, delta=1e-12, phi=0, q=q, method="fd", normalize=True):
    """
    Computes and plots dP/dΩ(θ) compared to sin²θ at a fixed radius.

    Parameters:
        radius: observation radius (cm)
        times: array of time samples over full period
        n_theta: number of θ points
        phi: azimuthal angle (fixed)
        method: "exact" or "fd"
        normalize: whether to normalize both curves to max = 1
    """
    theta_vals = np.linspace(0, np.pi, n_theta)
    dP_dOmega = np.zeros(n_theta)
    r = np.linalg.norm(r_obs)

    for i, theta in enumerate(theta_vals):
        S_dot_r = em.compute_S_dot_r(time_array=times, r_obs=r, r_o_func=r_o, theta=theta, phi=phi, delta=delta, q=q, method=method)
        dP_dOmega[i] = np.mean(S_dot_r) * np.power(r, 2)

    sin2_theta = np.sin(theta_vals)**2
    
    if normalize:
        dP_dOmega /= np.max(dP_dOmega)
        sin2_theta /= np.max(sin2_theta)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(theta_vals, dP_dOmega, label=r"Numerical $\frac{dP}{d\Omega}(\theta)$", linewidth=2.5)
    plt.plot(theta_vals, sin2_theta, '--', label=r"Theoretical $\sin^2(\theta)$", linewidth=2.5)
    plt.xlabel(r"$\theta$ (rad)", fontsize=12)
    plt.ylabel(r"Normalized $\frac{dP}{d\Omega}$", fontsize=12)
    plt.title(r"Angular Distribution of Average Power vs $\sin^2\theta$", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

#plot_angular_power_distribution_compare(r_obs=np.array([100,0,0]), r_o=r_o, times=times_test, n_theta=100, delta=1e-12, phi=0, q=q, method="exact", normalize=True)

def plot_residul_vs_theta(radius_list, times, r_o=r_o, n_theta=100, phi=0, delta=1e-12, q=q,method="exact", normalize=True):
    """
    Plots the residual (Numerical - sin²θ) for different radii to show onset of radiation zone.

    Parameters:
        radius_list: list of radii to test
        times: array of time samples
        n_theta: number of θ points
        phi: azimuthal angle (fixed)
        method: "exact" or "fd"
        normalize: whether to normalize curves
    """

    theta_vals = np.linspace(0, np.pi, n_theta)
    residuals = np.zeros((len(radius_list), n_theta))
    sin2_theta = np.sin(theta_vals)**2

    for j, radius in enumerate(radius_list):
        dP_dOmega = np.zeros(n_theta)

        for i, theta in enumerate(theta_vals):
            S_dot_r = em.compute_S_dot_r(time_array=times, r_obs=radius, r_o_func=r_o, theta=theta, phi=phi, delta=delta, q=q, method=method)
            dP_dOmega[i] = np.mean(S_dot_r) * np.power(radius, 2)
            #dP_dOmega[i] = em.angular_power_distribution(r_obs=radius, r_o_func=r_o, time_array=times, delta=delta, theta=theta, phi=phi, q=q, method=method)

        if normalize:
            dP_dOmega /= np.max(dP_dOmega)
            sin2_theta /= np.max(sin2_theta)

        residuals[j] = dP_dOmega - sin2_theta

    # Plot
    plt.figure(figsize=(6,4))
    for j, radius in enumerate(radius_list):
        plt.plot(theta_vals, residuals[j], label=f"Radius: {radius:.1f} cm", linewidth=2.5)

    plt.xlabel(r"$\theta$ (rad)", fontsize=12)
    plt.ylabel(r"Residual $|\frac{dP}{d\Omega} - \sin^2(\theta)|$",fontsize=12)
    plt.title("Residual vs θ for Different Radii", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

radius_list = np.array([5,10,50,100,200])  # Radii from 1 cm to 20 cm
#plot_residul_vs_theta(radius_list, times=times_test, r_o=r_o, n_theta=50, phi=0, delta=1e-12, q=q, method="exact", normalize=True)

def plot_max_residual_vs_radius(radius_list, times, r_o=r_o, n_theta=100, phi=0, delta=1e-12, q=q, normalize=True,method="exact"):
    """
    Computes and plots the maximum residual (Numerical - sin²θ) for different radii.

    Parameters:
        radius_list: list of radii to test
        times: array of time samples
        n_theta: number of θ points
        phi: azimuthal angle (fixed)
        method: "exact" or "fd"
    """

    theta_vals = np.linspace(0, np.pi, n_theta)
    max_residuals = np.zeros(len(radius_list))
    sin2_theta = np.sin(theta_vals)**2

    for j, radius in enumerate(radius_list):
        dP_dOmega = np.zeros(n_theta)

        for i, theta in enumerate(theta_vals):
            S_dot_r = em.compute_S_dot_r(time_array=times, r_obs=radius, r_o_func=r_o, theta=theta, phi=phi, delta=delta, q=q, method=method)
            dP_dOmega[i] = np.mean(S_dot_r) * np.power(radius, 2)

        if normalize:
            dP_dOmega /= np.max(dP_dOmega)
            sin2_theta /= np.max(sin2_theta)

        max_residuals[j] = np.max(np.abs(dP_dOmega - sin2_theta))

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(radius_list, max_residuals, marker='o', linewidth=2.5)
    plt.xlabel("Radius (cm)", fontsize=12)
    plt.ylabel("Max Residual", fontsize=12)
    plt.title("Max Residual vs Radius", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

radius_list_max = np.linspace(1, 200, 20)  # Radii from 1 cm to 200 cm
#plot_max_residual_vs_radius(radius_list_max, times=times_test, r_o=r_o, n_theta=120, phi=0, delta=1e-12, q=q, normalize=True, method="exact")


def analyze_Newtonian_limit(ratio_list, r_o=r_o, radius=100, times=None, l =1, c=c, n_theta=100, q=1, method="exact"):
    """
    Analyzes the Newtonian limit and its breakdown by varying ω.

    Parameters:
        w_list: list of angular frequencies ω to test
        l: amplitude of oscillation (cm)
        radius: observer radius (cm)
        times: array of times to evaluate over
        n_theta: number of theta points for angular distribution
        delta: finite differencing step size
        q: charge (esu)
        method: "exact" or "fd"
    """

    if times is None:
        times = np.linspace(0, 2 * np.pi / (0.01 * c / l), 300)  # Default: slowest period, 2 cycles

    r_obs = np.array([radius, 0, 0])  # observation point along x-axis
    #print(np.shape(ratio_list))
    fig, axs = plt.subplots(np.shape(ratio_list)[0], 3, figsize=(14, 8))

    theta_vals = np.linspace(0, np.pi, n_theta)
    sin2_theta = np.sin(theta_vals)**2
    sin2_theta /= np.max(sin2_theta)  # Normalize

    for idx, ratio in enumerate(ratio_list):
        w = ratio * c / l
        delta = 1 / (100 * w)  # Adjust delta based on w
        
        #times = np.linspace(0, 10 * np.pi / w, 300)

        # Define r_o for current w
        def r_o_specific(t):
            t = np.asarray(t).item()
            return np.array([0, 0, l * np.cos(w * t)])

        # --- E(t) and B(t) ---
        E_t, B_t = em.compute_E_B_vs_time(r_obs=r_obs, times=times, r_o_func=r_o_specific, delta=delta, q=q, method=method)
        row, col = idx , 0

        axs[row, col].plot(times, E_t, label=r"$|E(t)|$", color='blue')
        axs[row, col].set_title(r"$|\mathbf{E}(t)|$ for $\omega = %.1e$ rad/s, i.e., $\beta = %.2f$" % (w, ratio))
        axs[row, col].set_xlabel("Time (s)", fontsize=10)
        axs[row, col].set_ylabel(r"$|\mathbf{E}(t)|$", fontsize=10)
        axs[row, col].grid(True)

        col = 1
        axs[row, col].plot(times, B_t, label=r"$|B(t)|$", color='red')
        axs[row, col].set_title(r"$|\mathbf{B}(t)|$ for $\omega = %.1e$ rad/s, i.e., $v/c = %.2f$" % (w, ratio))
        axs[row, col].set_xlabel("Time (s)", fontsize=10)
        axs[row, col].set_ylabel(r"$|\mathbf{B}(t)|$", fontsize=10)
        #axs[row, col].legend()
        axs[row, col].grid(True)
        

        # --- dP/dΩ(θ) ---
        dP_dOmega = np.zeros(n_theta)
        for i, theta in enumerate(theta_vals):
            S_dot_r = em.compute_S_dot_r(time_array=times, r_obs=radius, r_o_func=r_o_specific, theta=theta, phi=0, delta=delta, q=q, method=method)
            dP_dOmega[i] = np.mean(S_dot_r) * radius**2
        
        dP_dOmega /= np.max(dP_dOmega)

        col = 2
        axs[row, col].plot(theta_vals, dP_dOmega, label="Numerical dP/dΩ", linewidth=2)
        axs[row, col].plot(theta_vals, sin2_theta, '--', label=r"Theoretical $\sin^2\theta$", linewidth=2)
        axs[row, col].set_title(f"Angular Distribution for ω = {w:.1e} rad/s")
        axs[row, col].set_xlabel(r"$\theta$ (rad)", fontsize=10)
        axs[row, col].set_ylabel(r"Normalized $dP/d\Omega$", fontsize=10)
        axs[row, col].legend()
        axs[row, col].grid(True)

    plt.tight_layout()
    plt.show()

#Example w values to test
ratio_list = [0.01, 0.3, 1.0]  # slow → relativistic

#times_test = np.linspace(0, 2 * np.pi / (0.01 * c / l), 300)  # Use period of slowest ω

# Call the function
#analyze_Newtonian_limit(ratio_list, r_o=r_o, radius=100, n_theta=100, q=q, method="exact")

def plot_dP_dw_spectrum():
    """
    Plots the spectrum of dP/dΩ(ω) for a fixed radius.
    """
    freq, dP_dw_dOmega = em.compute_dP_dw(r_obs=10, r_o_func=r_o, delta=1e-12, n_freq=1000, q=q)

R=1.0 

def r_o_circular(t, R=1.0, w=0.99*c/R):
    """
    Returns the position of the charge at time t performing circular motion in the xy-plane.
    """
    t = np.asarray(t).item()  # forces t to be a scalar
    return np.array([R * np.cos(w * t), R * np.sin(w * t), 0])

##getting stuff for this circular motion
x_cir = np.linspace(-10, 10, 200)
y_cir = np.linspace(-10, 10, 200)

T_cir =  2* np.pi / w  # Oscillation period
times_test_cir = np.linspace(0, 2*T, 100)  # 2 periods


#plot_retarded_time_xz_plane(t_test, x_cir, y_cir,r_o_func=r_o_circular)
#plot_retarded_time_xz_plane(t_test, x_cir, y_cir,r_o_func=r_o_circular)

plot_E_B_vs_time(r_obs=np.array([100,0,0]), times=times_test_cir, r_o_func=r_o_circular, delta=1e-16, method='exact')
#plot_S_dot_r_vs_theta(radial_distance=10, times=times_test, r_o_func=r_o_circular, theta_vals=np.array([0, np.pi/4, np.pi/2]), delta=1e-12, phi=0, method='exact')
#plot_angular_power_distribution_compare(r_obs=np.array([10,0,0]), r_o=r_o_circular, times=times_test, n_theta=100, delta=1e-12, phi=0, q=q, method="exact", normalize=True)
radius_list_circular = np.array([5,10,50,100,200])  # Radii from 1 cm to 20 c
#plot_residul_vs_theta(radius_list_circular, times=times_test, r_o=r_o_circular, n_theta=50, phi=0, delta=1e-12, q=q, method="exact", normalize=True)
radius_list_max_circular = np.linspace(1, 250, 10)  # Radii from 1 cm to 200 cm
#plot_max_residual_vs_radius(radius_list_max_circular, times=times_test, r_o=r_o_circular, n_theta=120, phi=0, delta=1e-12, q=q, normalize=True, method="exact")

def analyze_Newtonian_limit_circular(ratio_list, r_o=r_o_circular, radius=100, times=None, l =1, c=c, n_theta=100, q=1, method="exact"):
    """
    Analyzes the Newtonian limit and its breakdown by varying ω.

    Parameters:
        w_list: list of angular frequencies ω to test
        l: amplitude of oscillation (cm)
        radius: observer radius (cm)
        times: array of times to evaluate over
        n_theta: number of theta points for angular distribution
        delta: finite differencing step size
        q: charge (esu)
        method: "exact" or "fd"
    """

    if times is None:
        times = np.linspace(0, 2 * np.pi / (0.01 * c / l), 300)  # Default: slowest period, 2 cycles

    r_obs = np.array([radius, 0, 0])  # observation point along x-axis
    #print(np.shape(ratio_list))
    fig, axs = plt.subplots(np.shape(ratio_list)[0], 3, figsize=(14, 8))

    theta_vals = np.linspace(0, np.pi, n_theta)
    sin2_theta = np.sin(theta_vals)**2
    sin2_theta /= np.max(sin2_theta)  # Normalize

    for idx, ratio in enumerate(ratio_list):
        w = ratio * c / l
        delta = 1 / (100 * w)  # Adjust delta based on w
        
        times = np.linspace(0, 4 * np.pi / w, 300)

        # Define r_o for current w
        def r_o_specific(t):
            t = np.asarray(t).item()
            return r_o_circular(t, R=1.0, w=w)

        # --- E(t) and B(t) ---
        E_t, B_t = em.compute_E_B_vs_time(r_obs=r_obs, times=times, r_o_func=r_o_specific, delta=delta, q=q, method=method)
        row, col = idx , 0

        axs[row, col].plot(times, E_t, label=r"$|E(t)|$", color='blue')
        axs[row, col].set_title(r"$|\mathbf{E}(t)|$ for $\omega = %.1e$ rad/s, i.e., $\beta = %.2f$" % (w, ratio))
        axs[row, col].set_xlabel("Time (s)", fontsize=10)
        axs[row, col].set_ylabel(r"$|\mathbf{E}(t)|$", fontsize=10)
        axs[row, col].grid(True)

        col = 1
        axs[row, col].plot(times, B_t, label=r"$|B(t)|$", color='red')
        axs[row, col].set_title(r"$|\mathbf{B}(t)|$ for $\omega = %.1e$ rad/s, i.e., $v/c = %.2f$" % (w, ratio))
        axs[row, col].set_xlabel("Time (s)", fontsize=10)
        axs[row, col].set_ylabel(r"$|\mathbf{B}(t)|$", fontsize=10)
        #axs[row, col].legend()
        axs[row, col].grid(True)
        

        # --- dP/dΩ(θ) ---
        dP_dOmega = np.zeros(n_theta)
        for i, theta in enumerate(theta_vals):
            S_dot_r = em.compute_S_dot_r(time_array=times, r_obs=radius, r_o_func=r_o_specific, theta=theta, phi=0, delta=delta, q=q, method=method)
            dP_dOmega[i] = np.mean(S_dot_r) * radius**2
        
        dP_dOmega /= np.max(dP_dOmega)

        col = 2
        axs[row, col].plot(theta_vals, dP_dOmega, label="Numerical dP/dΩ", linewidth=2)
        axs[row, col].plot(theta_vals, sin2_theta, '--', label=r"Theoretical $\sin^2\theta$", linewidth=2)
        axs[row, col].set_title(f"Angular Distribution for ω = {w:.1e} rad/s")
        axs[row, col].set_xlabel(r"$\theta$ (rad)", fontsize=10)
        axs[row, col].set_ylabel(r"Normalized $dP/d\Omega$", fontsize=10)
        axs[row, col].legend()
        axs[row, col].grid(True)

    plt.tight_layout()
    plt.show()

#analyze_Newtonian_limit_circular(ratio_list, r_o=r_o, radius=100, n_theta=100, q=q, method="exact")
