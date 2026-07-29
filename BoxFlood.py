# Box model written by Neosha after Brinkerhoff et al. (2016)
# This is for a version where we set the parameters that are inverted for in Brinkerhoff et al. 
# There will be another version that actually carries out the MCMC inversion given a glacier velocity
# Created July 29, 2026
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

class BoxFlood:
    
    def __init__(self):
        self.A0_hat = 0.9
        self.P0_hat = 0.1
        self.k_hat  = 0.44 # velocity parameter
        self.r_hat  = 0.02
        self.Pi     = 0.09 #0.44   # was 0.09
        self.Psi    =  0.018 #0.61   # was 0.018
        self.Chi    = 0.5   # 3.41 # was 0.11  
        self.gamma  =  0.4 # 0.4    # was 0.22 #  <--- model stability is sensitive to this! 
        self.alpha  = 5/4   # was 5/4 # closure and opening rates?
        self.beta   = 3/2   # was 3/2

    def rhs(t, state, Q_in_func, k_hat, gamma, psi, r_hat, chi, pi, alpha, beta, n=3):

        A_hat, P_hat = state
    
        # Enforce physical bounds
        A_hat = max(A_hat, 0.0)
        Phat_upper_bound = 10 # by default, 1.0: when P_hat = 1, overburden, 0 = atmospheric pressure
        P_hat = np.clip(P_hat, 0.0, Phat_upper_bound)
    
        eps = 1e-6
        effective_pressure = max(Phat_upper_bound - P_hat, eps)  # prevents singularity
        
        # Equation 7
        sliding_opening = k_hat/((1-P_hat)**gamma)
        melting_opening = psi * r_hat * A_hat**alpha * P_hat**beta
        creep_closure = A_hat * (1 - P_hat)**n
    
        dA_dt = sliding_opening + melting_opening - creep_closure
    
        # Equation 8
        dP_dt = chi*(Q_in_func(t) - r_hat * A_hat**alpha * P_hat**(beta-1) - pi*dA_dt)
    
        return dA_dt, dP_dt

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
            max_step=0.05,
            rtol=1e-6,
            atol=1e-8,
        )
    
        # Calculate values
        A_hat = solution.y[0]
        P_hat = np.clip(solution.y[1], 0.0, 1.0)
        Q_out = r_hat * A_hat**alpha * P_hat**(beta-1)
        cavitation = k_hat/(1-P_hat)**gamma
        #print(solution.status)
    
        
        return solution.t, A_hat, P_hat, Q_out, cavitation

    def make_synthetic_Q_in(t_flood, flood_magnitude, flood_width, floodtype):
        """
        t_flood (int): day on which the flood occurs
        flood_magnitude (float): maximum magnitude of the rising part of the flood
        flood_width (float): duration of flood
        floodtype (str): "norm" = flood takes sinusoidal shape, "long" = long leadup flood
        """
        seasonal = 0.5 # Background value
        def Q_in_normal(t):
            #seasonal = 0.5 + 0.5 * (t / 50.0)           # slow ramp up
            diurnal  = 0.4 * np.sin(2 * np.pi * t)       # ~1 nondim day period
            flood    = flood_magnitude * np.exp(
                           -((t - t_flood) ** 2) / (2 * flood_width ** 2)) # normal curve centered about mean t_flood
    
            #total_Q = seasonal+diurnal+flood
            total_Q = seasonal + flood
            return max(total_Q, 0.0)
    
        def Q_in_longleadup(t):
             # Find the flood definition from matlab SHAKTI and put in here.
            falling_width = flood_width/10
    
            a = flood_magnitude/flood_width
    
            if t > t_flood and t <= t_flood + flood_width:
                Q = seasonal + a*(t-t_flood)
            elif t > t_flood + flood_width and t <= t_flood + flood_width + falling_width:
                b = -flood_magnitude/falling_width
                Q = seasonal + b*(t-(t_flood + flood_width)) + flood_magnitude
            else:
                Q = seasonal
             
            return max(Q, 0.0)
    
        if floodtype=="norm":
            return Q_in_normal
        if floodtype=="long":
            return Q_in_longleadup
        else:
            print("invalid flood type: returning normal")
            return Q_in_normal
        #return Q_in_longleadup


    