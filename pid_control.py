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


"""comparing two pid control agents with 1000 sampled initial states"""
@dataclass
class PidAgent:
    K_cA: float
    K_h: float


def sample_initial_state():
    return np.maximum(np.random.normal(loc=[0.8778252, 51.34660837, 0.659], scale=[0.25, 25, 0.2]), 0)


def get_error(agent, x0):
    """a temporary solution to unexpected error"""
    try:
        _, _, _, error = get_pid_simulation_data(agent.K_cA, agent.K_h, x0=x0)
    except:
        error = 1000
    return error if error < 1000 else 1000 # clip error

pid_agent1 = PidAgent(100, 10)
pid_agent2 = PidAgent(100, 0.2)
pid_errors = [[], []]
for _ in tqdm(range(1000)):
    x0 = sample_initial_state()
    for e, agent in enumerate([pid_agent1, pid_agent2]):
        pid_errors[e].append(get_error(agent, x0))

print("pid agent 1:", np.mean(pid_errors[0]), np.std(pid_errors[0]))
print("pid agent 2:", np.mean(pid_errors[1]), np.std(pid_errors[1]))
# the result may look like:
#pid agent 1: 28.38316702587552 138.20721123500766
#pid agent 2: 22.048147101429635 123.98955893515765


"""A Naive Baseline"""
"""train a linear regression model to decide the best Ks for PID control based on initial state"""
"""collect training data"""
from sklearn.linear_model import LinearRegression as LR
X = [] # initial states as features
Y = [] # control input as target
print("start collecting training samples...")
for _ in tqdm(range(100)):
    """collecting 100 X Y samples"""
    x0 = sample_initial_state()
    X.append(x0)
    """some arbitrary sampling for Ks, and select the best"""
    best_kca, best_kh, best_score = None, None, None
    for kca in np.random.randint(10, 100, 10):
        for kh in np.random.uniform(0, 10, 10):
            agent = PidAgent(kca, kh)
            error = get_error(agent, x0)
            if best_score is None or error < best_score:
                best_score = error
                best_kca, best_kh = kca, kh
    Y.append([best_kca, best_kh])

mdl = LR()
mdl.fit(np.array(X), np.array(Y))

"""test model performance"""
errors = []
for _ in tqdm(range(1000)):
    x0 = sample_initial_state()
    pred = mdl.predict(x0.reshape(1, -1))[0]
    agent = PidAgent(*pred)
    error = get_error(agent, x0)
    errors.append(error)

print("linear regression agent mean and std:", np.mean(errors), np.std(errors))
# the result may look like this:
#linear regression agent mean and std: 16.26030667876133 106.02515077559765
# so better than the two random ones!!!