import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import warnings
import matplotlib.animation as animation

# --- LGL SBP Class ---
class LGL_SBP:
    """
    Computes Legendre–Gauss–Lobatto nodes, quadrature weights, and the
    differentiation matrix D on the reference element [-1,1].
    Also computes the associated SBP norm matrix P and operator Q = P @ D.
    """
    def __init__(self, p):
        """Initializes LGL nodes, weights, and matrices for polynomial degree p."""
        if p < 1: raise ValueError("Polynomial degree p must be at least 1.")
        self.p = p
        self.num_nodes = p + 1
        self.nodes, self.weights = self._compute_nodes_weights() # xi in [-1, 1]
        self.D = self._differentiation_matrix() # d/dxi matrix
        self.P = np.diag(self.weights)  # Norm matrix P on [-1, 1]
        self.Q = self.P @ self.D        # SBP operator Q = P @ D
        self._verify_sbp_property()     # Check if Q + Q.T = B

    @staticmethod
    def legendre_poly_coeffs(p):
        """Computes coefficients of the Legendre polynomial of degree p."""
        if p == 0: return np.array([1.0]);
        if p == 1: return np.array([1.0, 0.0])
        poly_dict = {}
        for k in range(p // 2 + 1):
            power = p - 2 * k
            try: coeff = ((-1)**k * math.comb(p, k) * math.comb(2*p - 2*k, p)) / (2**p); poly_dict[power] = coeff
            except ValueError: pass # Handles cases where math.comb arguments are invalid
        return np.array([poly_dict.get(power, 0) for power in range(p, -1, -1)])

    @classmethod
    def legendre_poly(cls, p):
        """Returns a numpy.poly1d object for the Legendre polynomial."""
        return np.poly1d(cls.legendre_poly_coeffs(p))

    def _compute_nodes_weights(self):
        """Computes LGL nodes and weights for degree p."""
        if self.p == 0: return np.array([-1.0]), np.array([2.0])
        if self.p == 1: return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

        P_poly = self.legendre_poly(self.p); dP = P_poly.deriv()
        interior_nodes_complex = np.roots(dP.coeffs) # Roots of derivative P'_p
        interior_nodes = np.sort(interior_nodes_complex[np.isreal(interior_nodes_complex)].real)
        interior_nodes = interior_nodes[(interior_nodes > -1.0+1e-12) & (interior_nodes < 1.0-1e-12)] # Filter nodes strictly within (-1, 1)
        nodes = np.concatenate(([-1.0], interior_nodes, [1.0]))

        # Calculate weights using standard formula, handling potential precision issues
        P_vals_sq = P_poly(nodes)**2
        weights = np.zeros_like(nodes)
        denom = self.p * (self.p + 1) * P_vals_sq
        non_zero_denom_indices = np.abs(denom) > 1e-15
        weights[non_zero_denom_indices] = 2.0 / denom[non_zero_denom_indices]

        # Explicitly set known endpoint weights for robustness
        weights[0] = 2.0 / (self.p * (self.p + 1))
        weights[-1] = 2.0 / (self.p * (self.p + 1))

        if not np.isclose(np.sum(weights), 2.0):
             warnings.warn(f"Sum LGL weights={np.sum(weights):.4f} != 2.0.")
        return nodes, weights

    def _differentiation_matrix(self):
        """Computes the LGL differentiation matrix using barycentric formula."""
        x = self.nodes; N = len(x)
        if N <= 1: return np.zeros((N,N))

        D = np.zeros((N, N)); b = np.ones(N) # Barycentric weights (scaled)
        for j in range(N):
             prod = 1.0
             for k in range(N):
                  if j != k: prod *= (x[j] - x[k])
             if abs(prod) < 1e-15: # Avoid division by zero
                  warnings.warn(f"Zero product in barycentric weights at node {j}"); b[j] = 1e15
             else: b[j] = 1.0 / prod

        for i in range(N):
            sum_row = 0.0
            for j in range(N):
                if i != j:
                    term = (b[j] / b[i]) / (x[i] - x[j])
                    D[i, j] = term
                    sum_row += term
            D[i, i] = -sum_row # Diagonal entries ensure D @ constant_vector = 0
        return D
    
    def _verify_sbp_property(self):
        """Checks if the SBP property Q + Q.T = diag([-1, 0,...,0, 1]) holds."""
        B = np.diag([-1.0] + [0.0]*(self.p-1) + [1.0])
        if not np.allclose(self.Q + self.Q.T, B, atol=1e-12):
             max_dev = np.max(np.abs((self.Q + self.Q.T) - B))
             warnings.warn(f"LGL Q matrix SBP property deviation: {max_dev:.2e}")

# --- RK4 Time Stepper ---
def rk4_step(rhs_func, t, y, dt, *args):
    """Performs a single step of the classic Runge-Kutta 4th order method."""
    k1 = dt * rhs_func(t, y, *args)
    k2 = dt * rhs_func(t + 0.5 * dt, y + 0.5 * k1, *args)
    k3 = dt * rhs_func(t + 0.5 * dt, y + 0.5 * k2, *args)
    k4 = dt * rhs_func(t + dt, y + k3, *args)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

# --- RHS Function ---
def viscous_burgers_rhs_lgl_scaled(t, U, D1_phys, D2_phys, H_phys_inv_diag,
                                   nu, a, c, L_left, L_right,
                                   bc_func_L, bc_func_R):
    """Computes the RHS of the semi-discretized viscous Burgers' equation
       using scaled LGL operators and SAT boundary conditions."""
    u_x = D1_phys @ U
    advection_term = -U * u_x
    u_xx = D2_phys @ U
    diffusion_term = nu * u_xx

    g_prime_L_t = bc_func_L(t, nu, a, c, L_left)
    g_prime_R_t = bc_func_R(t, nu, a, c, L_right)

    SAT_viscous = np.zeros_like(U)
    SAT_viscous[0] = nu * H_phys_inv_diag[0] * (-g_prime_L_t) # Left boundary SAT term
    SAT_viscous[-1] = nu * H_phys_inv_diag[-1] * (g_prime_R_t) # Right boundary SAT term

    dUdt = advection_term + diffusion_term + SAT_viscous
    return dUdt

# --- Generalized Problem Definitions ---
def exact_solution_viscous_gen(x, t, nu, a, c):
    """Computes the generalized exact solution to the viscous Burgers' equation."""
    arg = a * (x - c * t) / (2.0 * nu)
    clip_val = 30 # Clip argument to tanh/cosh to prevent potential overflow
    arg = np.clip(arg, -clip_val, clip_val)
    tanh_val = np.tanh(arg)
    return c - a * tanh_val

def initial_condition_gen(x, nu, a, c):
    """Computes the initial condition from the exact solution at t=0."""
    return exact_solution_viscous_gen(x, 0.0, nu, a, c)

def bc_neumann_left_gen(t, nu, a, c, L_left):
    """Computes the exact time-dependent Neumann BC u_x at x=L_left."""
    arg = a * (L_left - c * t) / (2.0 * nu)
    clip_val = 30
    arg = np.clip(arg, -clip_val, clip_val)
    cosh_val = np.cosh(arg)
    # sech^2(y) = 1 / cosh^2(y)
    # u_x = -a^2/(2*nu) * sech^2(arg)
    sech_sq = 1.0 / (cosh_val**2) if not np.isclose(cosh_val, 0.0) else 0.0
    return - (a**2 / (2.0 * nu)) * sech_sq

def bc_neumann_right_gen(t, nu, a, c, L_right):
    """Computes the exact time-dependent Neumann BC u_x at x=L_right."""
    arg = a * (L_right - c * t) / (2.0 * nu)
    clip_val = 30
    arg = np.clip(arg, -clip_val, clip_val)
    cosh_val = np.cosh(arg)
    sech_sq = 1.0 / (cosh_val**2) if not np.isclose(cosh_val, 0.0) else 0.0
    return - (a**2 / (2.0 * nu)) * sech_sq

# --- Simulation Setup Function ---
def setup_simulation(p, L_left, L_right, nu, a, c, CFL_adv, CFL_diff):
    """Sets up the LGL grid, scaled operators, IC, and initial dt."""
    print("--- Setting up Simulation ---")
    domain_length = L_right - L_left
    if domain_length <= 0: raise ValueError("L_right must be greater than L_left.")

    lgl = LGL_SBP(p)
    xi_nodes = lgl.nodes
    D1_ref = lgl.D
    weights_ref = lgl.weights
    P_diag_ref = weights_ref
    Pinv_diag_ref = 1.0 / weights_ref

    # Map nodes and Scale Operators
    x_lgl_nodes_phys = L_left + domain_length * (xi_nodes + 1.0) / 2.0
    deriv_scale = 2.0 / domain_length
    D1_phys = D1_ref * deriv_scale
    D1T_ref = D1_ref.T
    term1 = np.diag(Pinv_diag_ref); term2 = D1T_ref
    term3 = np.diag(P_diag_ref); term4 = D1_ref
    D2_ref = -term1 @ term2 @ term3 @ term4 # D2 on reference element
    D2_phys = D2_ref * (deriv_scale**2)     # Scale D2 for physical domain
    H_phys_inv_diag = Pinv_diag_ref * deriv_scale # Scaled inverse norm diagonal for SAT

    U0 = initial_condition_gen(x_lgl_nodes_phys, nu, a, c)

    # Estimate initial Time Step based on minimum physical spacing
    min_dx_phys = np.min(np.diff(x_lgl_nodes_phys))
    max_speed_initial = np.max(np.abs(U0)); max_speed_initial = max(max_speed_initial, 1e-9)
    dt_adv_est = CFL_adv * min_dx_phys / max_speed_initial
    dt_diff_est = CFL_diff * min_dx_phys**2 / nu
    initial_dt = min(dt_adv_est, dt_diff_est)

    print(f"Domain: [{L_left}, {L_right}] (Length: {domain_length:.2f})")
    print(f"Parameters: a={a}, c={c}, nu={nu}")
    print(f"Polynomial Degree p = {p} ({p+1} LGL nodes)")
    print(f"Min Physical Node Spacing = {min_dx_phys:.4e}")
    print(f"Initial time step dt = {initial_dt:.6e}")
    if dt_diff_est < dt_adv_est: print("!!! Diffusion stability likely limiting dt !!!")
    print("---------------------------------------------")

    return (x_lgl_nodes_phys, D1_phys, D2_phys, H_phys_inv_diag, weights_ref, U0,
            initial_dt, domain_length)

# --- Time Integration Function ---
def run_simulation(Tfinal, U0, initial_dt, x_lgl_nodes_phys,
                   D1_phys, D2_phys, H_phys_inv_diag,
                   nu, a, c, L_left, L_right,
                   CFL_adv, CFL_diff, save_every):
    """Performs the time integration using RK4."""
    print("--- Running Simulation ---")
    t = 0.0
    U = U0.copy()
    dt = initial_dt
    time_steps = [t]; solution_history = [U.copy()]; nsteps = 0

    while t < Tfinal:
        # Adjust dt if step would overshoot Tfinal
        if t + dt > Tfinal: dt = Tfinal - t

        # Adapt dt based on current CFL conditions
        min_dx = np.min(np.diff(x_lgl_nodes_phys))
        current_max_speed = np.max(np.abs(U)); current_max_speed = max(current_max_speed, 1e-9)
        dt_adv = CFL_adv * min_dx / current_max_speed
        dt_diff = CFL_diff * min_dx**2 / nu
        current_dt = min(dt_adv, dt_diff)
        # Prevent dt from growing excessively if stability condition temporarily relaxes
        dt = min(dt, current_dt) if nsteps > 0 else current_dt

        # Check for excessively small dt
        if dt < 1e-15:
            warnings.warn("Time step too small, stopping simulation prematurely.")
            break

        # Perform RK4 step
        U = rk4_step(viscous_burgers_rhs_lgl_scaled, t, U, dt,
                     D1_phys, D2_phys, H_phys_inv_diag,
                     nu, a, c, L_left, L_right,
                     bc_neumann_left_gen, bc_neumann_right_gen)
        t += dt
        nsteps += 1

        # Store solution history
        if nsteps % save_every == 0:
            if not time_steps or abs(t - time_steps[-1]) > 1e-12: # Avoid near duplicates
                 time_steps.append(t)
                 solution_history.append(U.copy())

    # Ensure final step is captured if needed
    if not np.isclose(time_steps[-1], Tfinal) and time_steps[-1] < Tfinal:
         time_steps.append(t)
         solution_history.append(U.copy())

    print(f"Simulation finished at t = {time_steps[-1]:.4f} after {nsteps} steps.")
    print(f"Stored {len(time_steps)} time levels.")
    return time_steps, solution_history

# --- Plotting Function ---
def plot_specific_times(x_lgl_nodes_phys, time_steps, solution_history, U0_plot,
                        target_plot_times, L_left, L_right, nu, a, c, p):
    """Creates a static plot comparing numerical and exact solutions at specified times."""
    print("\n--- Generating Static Plot ---")
    fig, ax = plt.subplots(figsize=(12, 7))

    num_plot_points_exact = 101
    x_plot_fine = np.linspace(L_left, L_right, num_plot_points_exact)

    plot_indices = []; actual_plot_times = []; valid_target_times = []
    time_steps_array = np.array(time_steps)
    for target_t in target_plot_times:
        if target_t > time_steps_array[-1] + 1e-9:
            warnings.warn(f"Plot time t={target_t} > sim end time {time_steps_array[-1]:.4f}. Skipping.")
            continue
        idx = np.argmin(np.abs(time_steps_array - target_t))
        plot_indices.append(idx)
        actual_plot_times.append(time_steps_array[idx])
        valid_target_times.append(target_t)

    print(f"Plotting times close to: {valid_target_times}")

    colors_exact = cm.viridis(np.linspace(0.1, 0.9, len(plot_indices))) # Use slightly adjusted range for visibility
    numerical_color = 'red'

    legend_exact_handles = []
    legend_num_combined = None

    for i, idx in enumerate(plot_indices):
        t_plot_actual = actual_plot_times[i]
        U_plot_numerical = solution_history[idx]
        U_exact_plot_fine = exact_solution_viscous_gen(x_plot_fine, t_plot_actual, nu, a, c)
        color_exact = colors_exact[i]
        time_label = f't={valid_target_times[i]:.2f}'

        line_exact, = ax.plot(x_plot_fine, U_exact_plot_fine,
                               linestyle='-', linewidth=2.0,
                               color=color_exact, label=f'Exact {time_label}')
        legend_exact_handles.append(line_exact)

        line_num, = ax.plot(x_lgl_nodes_phys, U_plot_numerical,
                             linestyle='--', linewidth=1.0, marker='o', markersize=3, # Slightly smaller markers
                             color=numerical_color, alpha=0.6, label=f'Num {time_label}')
        if legend_num_combined is None: legend_num_combined = line_num

    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(f"Viscous Burgers (nu={nu}, N={p+1}) on Domain [{L_left},{L_right}] at Specific Times - Solid (Exact), Dashed+Markers (Numerical)")
    ax.grid(True)
    ax.set_ylim(c - a - 0.3, c + a + 0.3) # Adjust y-limits based on parameters

    # Create Legend
    legend_handles = legend_exact_handles
    legend_labels = [f'Exact t = {valid_target_times[i]:.2f}' for i in range(len(valid_target_times))]
    if legend_num_combined is not None:
        legend_handles.append(legend_num_combined)
        legend_labels.append(f'Numerical (All Times)')
    ax.legend(handles=legend_handles, labels=legend_labels, fontsize='medium', loc='best')

    fig.tight_layout()
    plt.show()
    print("\nStatic plotting complete.")


# --- Animation Function ---
def create_animation(x_lgl_nodes_phys, time_steps, solution_history, U0_anim,
                     L_left, L_right, nu, a, c, p, interval_ms, filename=None):
    """Creates and displays or saves an animation of the simulation results."""
    print("\n--- Setting up Animation ---")
    fig_anim, ax_anim = plt.subplots(figsize=(10, 6))

    num_plot_points_exact = 101
    x_plot_fine = np.linspace(L_left, L_right, num_plot_points_exact)

    # Initial state plot elements
    U_exact_0 = exact_solution_viscous_gen(x_plot_fine, time_steps[0], nu, a, c)
    line_exact_anim, = ax_anim.plot(x_plot_fine, U_exact_0, '-', lw=1.5, color='blue', label='Exact')
    U_num_0 = solution_history[0]
    line_num_anim, = ax_anim.plot(x_lgl_nodes_phys, U_num_0, '--o', ms=5, lw=1.0, color='red', label='Numerical')
    time_text_anim = ax_anim.text(0.05, 0.9, f'Time = {time_steps[0]:.3f}', transform=ax_anim.transAxes, fontsize=12)

    # Configure plot appearance
    ax_anim.set_xlim(L_left, L_right)
    ax_anim.set_ylim(c - a - 0.3, c + a + 0.3)
    ax_anim.set_xlabel('x')
    ax_anim.set_ylabel('u(x, t)')
    ax_anim.set_title(f"Viscous Burgers Animation (nu={nu}, N={p+1}) on Domain [{L_left},{L_right}]")
    ax_anim.legend(loc='upper right')
    ax_anim.grid(True)
    fig_anim.tight_layout()

    # Animation update function
    def update_anim(i):
        line_num_anim.set_ydata(solution_history[i])
        current_t = time_steps[i]
        line_exact_anim.set_ydata(exact_solution_viscous_gen(x_plot_fine, current_t, nu, a, c))
        time_text_anim.set_text(f'Time = {current_t:.3f}')
        return line_num_anim, line_exact_anim, time_text_anim

    # Create and process animation
    ani = animation.FuncAnimation(fig_anim, update_anim, frames=len(solution_history),
                                  interval=interval_ms, blit=True, repeat=False)

    print(f"Animation frames: {len(solution_history)}")
    print(f"Animation interval: {interval_ms} ms")

    if filename:
        # Attempt to save the animation
        try:
            if filename.lower().endswith('.mp4'):
                ani.save(filename, writer='ffmpeg', fps=1000//interval_ms, dpi=150)
            elif filename.lower().endswith('.gif'):
                 ani.save(filename, writer='imagemagick', fps=1000//interval_ms)
            else:
                print(f"Unsupported animation format: {filename.split('.')[-1]}. Use .mp4 or .gif.")
                filename = None # Prevent closing plot if save failed due to format
            if filename: print(f"Animation saved to {filename}")
        except Exception as e:
            print(f"Error saving animation to {filename}: {e}.")
            print("Ensure ffmpeg (for mp4) or ImageMagick (for gif) is installed and in PATH.")
            filename = None # Prevent closing plot if save failed
        finally:
            if filename: plt.close(fig_anim) # Close figure only if save was attempted and potentially successful
            else: plt.show() # Display if not saving
    else:
        plt.show() # Display if filename is None

    print("\nAnimation processing complete.")


# --- Main Execution Block ---
def main():
    """Runs the viscous Burgers' equation simulation and generates outputs."""
    # --- User-Defined Parameters ---
    nu = 0.1          # Viscosity
    a = 1.0           # Amplitude parameter in IC/Exact solution
    c = 1.0           # Wave speed / DC offset parameter
    L_left = -0.8     # Left boundary of physical domain
    L_right = 1.6     # Right boundary of physical domain
    p = 40            # Polynomial degree for LGL nodes (p+1 points)
    Tfinal = 1.0      # Simulation end time
    CFL_adv = 0.05    # CFL number for advection term (adjust for stability)
    CFL_diff = 0.02   # CFL number for diffusion term (adjust for stability)
    save_every = 10   # Store history every 'save_every' steps (lower = more frames)

    # --- Output Control ---
    CREATE_STATIC_PLOT = True # Generate the static plot with specific time snapshots
    STATIC_PLOT_TIMES = np.linspace(0, Tfinal, 5).tolist() # Times for static plot

    CREATE_ANIMATION = True   # Generate the animation
    ANIMATION_INTERVAL_MS = 1 # Delay between animation frames (ms) - Lower is faster
    ANIMATION_FILENAME = None  # Set to 'burgers_lgl.mp4' or 'burgers_lgl.gif' to save, None to display

    # --- Setup Simulation ---
    (x_lgl_nodes_phys, D1_phys, D2_phys, H_phys_inv_diag, weights_ref, U0,
     initial_dt, domain_length) = setup_simulation(
         p, L_left, L_right, nu, a, c, CFL_adv, CFL_diff
     )

    # --- Run Simulation ---
    time_steps, solution_history = run_simulation(
        Tfinal, U0, initial_dt, x_lgl_nodes_phys,
        D1_phys, D2_phys, H_phys_inv_diag,
        nu, a, c, L_left, L_right,
        CFL_adv, CFL_diff, save_every
    )

    # --- Post-processing: Calculate Final Error ---
    U_final = solution_history[-1]
    t_final = time_steps[-1]
    U_exact_final_lgl = exact_solution_viscous_gen(x_lgl_nodes_phys, t_final, nu, a, c)
    # Calculate L2 error using the appropriate norm for the domain mapping
    error_l2_physical = np.sqrt(np.sum(weights_ref * (U_final - U_exact_final_lgl)**2 * (domain_length / 2.0)))
    print(f"\n--- Results at T = {t_final:.4f} ---")
    print(f"Physical L2 Error (LGL Norm): {error_l2_physical:.4e}")
    print(f"------------------------")

    # --- Generate Outputs ---
    if CREATE_STATIC_PLOT:
        plot_specific_times(
            x_lgl_nodes_phys, time_steps, solution_history, U0,
            target_plot_times = STATIC_PLOT_TIMES,
            L_left=L_left, L_right=L_right, nu=nu, a=a, c=c, p=p
        )

    if CREATE_ANIMATION:
        create_animation(
            x_lgl_nodes_phys, time_steps, solution_history, U0,
            L_left=L_left, L_right=L_right, nu=nu, a=a, c=c, p=p,
            interval_ms=ANIMATION_INTERVAL_MS,
            filename=ANIMATION_FILENAME
        )

# --- Script Execution Entry Point ---
if __name__ == "__main__":
    main()
