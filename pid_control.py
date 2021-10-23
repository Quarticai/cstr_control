from dataclasses import dataclass
from tqdm import tqdm
from models import *


XS = np.array([0.8778252, 51.34660837, 0.659])
US = np.array([26.85, 0.1])


def objective_func(x, xsp):
    return np.mean((x - xsp) ** 2 / np.maximum((x[0] - xsp[0]) ** 2, 1e-8))


def get_pid_simulation_data(K_cA, K_h, x0=np.array([0.95413317, 49.08828331,  0.53214008])):
    dt = 0.1  # sampling time
    # build reactor simulator
    reactor = ReactorModel(dt)
    reactor.build_reactor_simulator()
    Nx = reactor.Nx
    Nu = reactor.Nu
    xs = XS

    # build P-controller
    p_controller = PID(K_cA, K_h, XS, US)

    # simulation settings
    Nsim = 100
    x = np.zeros((Nsim + 1, Nx))
    x[0, :] = x0
    u = np.zeros((Nsim, Nu))

    # Setpoint data for plot
    xsp = np.zeros((Nsim + 1, Nx))
    xsp[0, :] = xs

    # simulate closed-loop system
    for t in range(Nsim):
        uopt = p_controller.solve(x[t, :])
        u[t, :] = uopt[:]
        x[t + 1, :] = reactor.step(x[t, :], u[t, :])
        xsp[t + 1, :] = xs

    return x, xsp, u, objective_func(x, xsp)


"""comparing two pid control agents with 100 sampled initial states"""
@dataclass
class PidAgent:
    K_cA: float
    K_h: float


pid_agent1 = PidAgent(100, 10)
pid_agent2 = PidAgent(100, 0.2)
pid_errors = [[], []]
for _ in tqdm(range(100)):
    x0 = np.maximum(np.random.normal(loc=[0.8778252, 51.34660837, 0.659], scale=[0.25, 25, 0.2]), 0)
    for e, agent in enumerate([pid_agent1, pid_agent2]):
        try:
            _, _, _, error = get_pid_simulation_data(agent.K_cA, agent.K_h, x0=x0)
        except:
            error = 100
        pid_errors[e].append(error)
print(pid_errors)
print("pid agent 1:", np.mean(pid_errors[0]), np.std(pid_errors[0]))
print("pid agent 2:", np.mean(pid_errors[1]), np.std(pid_errors[1]))
