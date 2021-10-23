# -*- coding: utf-8 -*-
import mpctools as mpc      # import mpctools: https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/
import numpy as np			# import numpy

class ReactorModel:
    
    def __init__(self, sampling_time):
        
        # Define model parameters
        self.q_in = .1        # m^3/min
        self.Tf = 76.85     # degrees C
        self.cAf = 1.0       # kmol/m^3
        self.r = .219       # m
        self.k0 = 7.2e10    # min^-1
        self.E = 8750       # K
        self.U = 54.94      # kg/min/m^2/K
        self.rho = 1000     # kg/m^3
        self.Cp = .239      # kJ/kg/K
        self.dH = -5e4      # kJ/kmol
        
        self.Nx = 3         # Number of state variables
        self.Nu = 2         # Number of input variables
        
        self.sampling_time = sampling_time      # sampling time or integration step
    
    # Ordinary Differential Equations (ODEs) described in the report i.e. Equations (1), (2), (3)
    def ode(self, x, u):
    
        c = x[0]        # c_A
        T = x[1]        # T
        h = x[2]        # h
        Tc = u[0]       # Tc
        q = u[1]        # q_out
        
        rate = self.k0*c*np.exp(-self.E/(T+273.15))  # kmol/m^3/min
        
        dxdt = [
            self.q_in*(self.cAf - c)/(np.pi*self.r**2*h) - rate, # kmol/m^3/min
            self.q_in*(self.Tf - T)/(np.pi*self.r**2*h) 
                        - self.dH/(self.rho*self.Cp)*rate
                        + 2*self.U/(self.r*self.rho*self.Cp)*(Tc - T), # degree C/min
            (self.q_in - q)/(np.pi*self.r**2)     # m/min
                ]
        return dxdt
    
    # builds a reactor using mpctools and casadi
    def build_reactor_simulator(self):
        self.simulator = mpc.DiscreteSimulator(self.ode, self.sampling_time, [self.Nx, self.Nu], ["x", "u"])
    
    # integrates one sampling time or time step and returns the next state
    def step(self, x, u):
        return self.simulator.sim(x,u)


class MPC:
    
    def __init__(self, Nt, dt, Q, R, P):
        self.Nt = Nt    # control and prediction horizons
        self.dt = dt    # sampling time. Should be the same as the step size in the model
        self.Q = Q      # weight on the state variables
        self.R = R      # weight on the input variables
        self.P = P      # terminal cost weight (optional). set to zero matrix with appropriate dimension
        self.Nx = 3     # number of state variables
        self.Nu = 2     # number of input variables
        
        self.xs = np.array([0.8778252,51.34660837,0.659])   # steady-state state values 
        self.us = np.array([26.85,0.1])     # steady state input values
        
    # Define stage cost
    def lfunc(self, x, u):
        dx = x[:self.Nx] - self.xs[:self.Nx]
        du = u[:self.Nu] - self.us[:self.Nu]
        return mpc.mtimes(dx.T,self.Q,dx) + mpc.mtimes(du.T,self.R,du)
    
    # define terminal weight
    def Pffunc(self, x):
        dx = x[:self.Nx] - self.xs[:self.Nx]
        return mpc.mtimes(dx.T,self.P,dx)
    
    # build the mpc controller using mpctools
    def build_controller(self):
        
        # stage and terminal cost in casadi symbolic form
        l = mpc.getCasadiFunc(self.lfunc, [self.Nx,self.Nu], ["x","u"], funcname="l")
        Pf = mpc.getCasadiFunc(self.Pffunc, [self.Nx], ["x"], funcname="Pf")
        
        model = ReactorModel(self.dt)
        
        # Create casadi function/model to be used in the controller
        ode_casadi = mpc.getCasadiFunc(model.ode, [self.Nx, self.Nu], ["x","u"], funcname="odef")

        # Set c to a value greater than one to use collocation
        contargs = dict(
                N = {"t":self.Nt, "x":self.Nx, "u":self.Nu, "c":3},
                verbosity=0,    # set verbosity to a number > 0 for debugging purposes otherwise use 0.
                l=l,
                x0=self.xs,
                Pf=Pf,
                ub = {"u":1.2*self.us[:self.Nu],"x":2*self.xs[:self.Nx]}, # Change upper bounds 
                lb = {"u":0.8*self.us[:self.Nu],"x":0.5*self.xs[:self.Nx]},	# Change lower bounds 
                guess = {
                        "x":self.xs[:self.Nx],
                        "u":self.us[:self.Nu]
                        },
                )
        
        # create MPC controller
        self.mpc = mpc.nmpc(f=ode_casadi,Delta=self.dt,**contargs)
    
    # solve mpc optimization problem at one time step
    def solve(self,x):
        # Update intial condition and solve control problem.
        self.mpc.fixvar("x",0,x) # Set initial value
        self.mpc.solve() # solve optimal control problem
        uopt = np.squeeze(self.mpc.var["u",0]) # take the first optimal input solution
        
        # if successful, save solution as initial guess for the next optimization problem
        if self.mpc.stats["status"] == "Solve_Succeeded":
            self.mpc.saveguess()
        
        # return solution status and optimal input
        return self.mpc.stats["status"], uopt
    
class PID:
    
    def __init__(self, K_cA, K_h, xs, us):
        
        self.K_cA = K_cA    # porportional gain for concentration
        self.K_h = K_h      # porportional gain for level
        
        self.xs = xs   # steady-state state values
        self.us = us     # steady state input values
        
    # this implements a simple proportional controller
    def solve(self, x):
        
        u = [
            self.K_cA*(x[0] - self.xs[0]) + self.us[0],
            self.K_h*(x[2] - self.xs[2]) + self.us[1]
            ]
        
        return np.maximum(np.array(u), 0)
        