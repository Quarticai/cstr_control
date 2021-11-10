"""1 step reinforce or vanilla policy gradient"""

from tqdm import tqdm
import torch
from models import *

"""env config"""
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


def sample_initial_state():
    return np.maximum(np.random.normal(loc=[0.8778252, 51.34660837, 0.659], scale=[0.25, 25, 0.2]), 0)


def get_error(k_ca, k_h, x0):
    """a temporary solution to unexpected error"""
    try:
        _, _, _, error = get_pid_simulation_data(k_ca, k_h, x0=x0)
    except:
        error = 1000
    return error if error < 1000 else 1000


"""agent config"""


class ReinforceAgent(torch.nn.Module):
    def __init__(self, obs_dim=3, act_dim=2, hidden_size=6):
        super(ReinforceAgent, self).__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = torch.nn.Sequential(torch.nn.Linear(obs_dim, hidden_size),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_size, act_dim))

    def forward(self, obs):
        mu = self.mu_net(obs)
        std = torch.maximum(torch.exp(self.log_std), torch.ones_like(self.log_std)*1e-4)
        return torch.distributions.normal.Normal(mu, std)


"""training starts here"""
agent = ReinforceAgent()
opt = torch.optim.Adam(agent.parameters())
for batch in tqdm(range(100)):
    states = []
    actions = []
    rewards = []
    """run the agent to collect data
       for policy gradient, we interact the environment to collect new data for each training epoch 
    """
    with torch.no_grad():
        for i in range(100):
            state = sample_initial_state()
            states.append(state)
            """the action distribution given by current state"""
            pi = agent(torch.from_numpy(np.array(state, dtype='float32').reshape(1, -1)))
            action = np.clip(pi.sample().numpy().flatten(), [0, 0], [100, 0.2]) # clip actions
            actions.append(action)
            rewards.append(-get_error(action[0], action[1], state)) # -error is reward
    """update agent with collected data"""
    rewards = (np.array(rewards) - np.mean(rewards))/np.std(rewards) # standardize rewards
    states = torch.from_numpy(np.array(states, dtype='float32'))
    actions = torch.from_numpy(np.array(actions, dtype='float32'))
    rewards = torch.from_numpy(np.array(rewards, dtype='float32'))
    """calculate log prob"""
    pi = agent(states)
    logp = pi.log_prob(actions).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    """gradient ascent"""
    loss = -(logp*rewards).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()

"""evaluate agent performance"""
errors = []
"""run the agent to collect data"""
with torch.no_grad():
    for _ in tqdm(range(1000)):
        state = sample_initial_state()
        pi = agent(torch.from_numpy(np.array(state, dtype='float32').reshape(1, -1)))
        action = np.clip(pi.sample().numpy().flatten(), [0, 0], [100, 0.2])  # clip actions
        errors.append(get_error(action[0], action[1], state))

print(np.mean(errors), np.std(errors))
# 16.583420051917695 106.9320081128052