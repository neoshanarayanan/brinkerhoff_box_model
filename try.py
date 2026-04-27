from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Now code up equations 7 and 8

def rhs(t, state, Q_in_func, k_hat, gamma, psi, r_hat, chi, pi, alpha, beta, n=3):

    A_hat, P_hat = state

    # Enforce physical bounds
    A_hat = max(A_hat, 0.0)
    P_hat = np.clip(P_hat, 0.0, 1.0)

    
    # Equation 7
    sliding_opening = k_hat/((1-P_hat)**gamma)
    melting_opening = psi * r_hat * A_hat**alpha * P_hat**beta
    creep_closure = A_hat * (1 - P_hat)**n

    dA_dt = sliding_opening + melting_opening - creep_closure

    # Equation 8 (differs from Claude version)
    dP_dt = chi*(Q_in_func(t) - r_hat * A_hat**alpha * P_hat**(beta-1) - pi*dA_dt)

    return dA_dt, dP_dt

# Initial values from MCMC
A0_hat = 0.9 # Initial cavity size (defined with MCMC)
P0_hat = 0.2 # Initial pressure (defined MCMC)
k_hat = 0.44 # Basal traction coefficient
r_hat = 0.02 # linear flux coefficient


Pi = 0.09 # From Bartholomaus et al. 
Psi = 0.018 # From Bartholomaus et al. 
Chi = 0.11 # Englacial storage coefficient from Bartholomaus
gamma = 0.22 # nonlinear dependence of sliding speed on effective pressure
alpha = 5/4
beta = 3/2

def solve(t_span, t_eval, Q_in_func):
    params = dict(
        Q_in_func=Q_in_func,
        k_hat=k_hat, gamma=gamma, psi=Psi, r_hat=r_hat,
        chi=Chi, pi=Pi, alpha=alpha, beta=beta, n=3
    )
    solution = solve_ivp(
        fun = lambda t, y: rhs(t, y, **params),  # y = state
        t_span = t_span,
        t_eval = t_eval,
        y0 = [A0_hat, P0_hat],
        method = "RK45", # Runge Kutta
    )

    # Calculate values
    A_hat = solution.y[0]
    P_hat = np.clip(solution.y[1], 0.0, 1.0)
    Q_out = r_hat * A_hat**alpha * P_hat**(beta-1)
    print(solution.status)

    
    return solution.t, A_hat, P_hat, Q_out

def make_synthetic_Q_in(t_flood=32.5, flood_magnitude=6.0, flood_width=0.5):
    """
    Simple synthetic forcing: diurnal oscillation on a slow seasonal ramp,
    plus a single outburst flood pulse, roughly mimicking the Kennicott 2006
    record used in the paper.

    All times are nondimensional.
    """
    def Q_in(t):
        seasonal = 0.5 + 0.5 * (t / 50.0)           # slow ramp up
        diurnal  = 0.4 * np.sin(2 * np.pi * t)       # ~1 nondim day period
        flood    = flood_magnitude * np.exp(
                       -((t - t_flood) ** 2) / (2 * flood_width ** 2))
        return max(seasonal + diurnal + flood, 0.0)

    return Q_in

t_start, t_end = 0.0, 52.0
t_eval = np.linspace(t_start, t_end, 5000)

Q_in_func = make_synthetic_Q_in()

times, A_hat, P_hat, Q_out = solve(
    t_span=(t_start, t_end),
    t_eval = t_eval,
    Q_in_func=Q_in_func
)

