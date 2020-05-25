import numpy as np
import matplotlib.pyplot as plt
import random


class Environment:
    """Used to represent gridworld environment
    
    # Arguments
        width: integer. The width of the grid
        height: integer. The height of the grid
        start: tuple. The start location on grid, (col, row)
        goal: tuple. The goal location on grid, (col, row)
        first_blocked_cells: list. Initial cells which the agent cannot pass
        next_blocked_cells: list. Updated blocked cells, optional.
    """

    def __init__(self, width, height, start,
                 goal, first_blocked_cells,
                 next_blocked_cells=None):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.first_blocked_cells = first_blocked_cells
        self.next_blocked_cells = next_blocked_cells

        self.blocked_cells = first_blocked_cells
        self.done = False
        # action2move maps action choice to direction, U, D, L & R
        self.action2move = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
    def reset(self):
        self.done = False

    def change_blocked_cells(self):
        self.blocked_cells = self.next_blocked_cells

    def reset_blocked_cells(self):
        self.blocked_cells = self.first_blocked_cells

    def print_grid(self):
        grid = np.zeros((self.height, self.width))
        grid[self.start[0]][self.start[1]] = 1
        grid[self.goal[0]][self.goal[1]] = 2
        for cell in self.blocked_cells:
            grid[cell[0]][cell[1]] = 3
        print(grid)
        
    def step(self, state, action):
        dy, dx = self.action2move[action]
        
        new_y = np.clip((state[0] + dy), 0, self.height-1)
        new_x = np.clip((state[1] + dx), 0, self.width-1)
        
        next_state = (new_y, new_x)
        if next_state == self.goal:
            self.done = True
            return 1, self.start
        elif next_state in self.blocked_cells:
            return 0, state
        else:
            return 0, next_state


class DynaQ:
    """DynaQ agent

    # Arguments
        env: Environment class instance
        planning_n: integer. Param to control sampling at each timestep
        alpha: float 0-1. Stepsize parameter
        epsilon: float 0-1. Parameter to control E-Greedy action selection
        gamma: float 0-1. Discount factor on expected future rewards
        n_steps: integer. Number of steps to run before returning results
    """

    def __init__(self, env, planning_n, n_steps, change_step, 
                 alpha=0.1, epsilon=0.1, gamma=0.95):
        self.env = env
        self.planning_n = planning_n
        self.n_steps = n_steps
        self.change_step = change_step
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.timestep = 0
        self.n_actions = 4
        # Model maps state-action pairs to next state and reward
        self.model = {}
        # Qsa is Q-estimate table for each state-action pair
        self.Qsa = np.zeros((self.env.height, self.env.width, self.n_actions))
    
    def __repr__(self):
        return 'DynaQ'

    def print_optimal_path(self):
        # Helper function for debugging
        optimal_path = []
        state = self.env.start
        while True:
            optimal_path.append(state)
            action = np.argmax(self.Qsa[state[0], state[1]])
            _, state = self.env.step(state, action)
            if state == self.env.goal:
                break
        optimal_grid = []
        for i in range(self.env.height):
            optimal_grid.append([])
            for j in range(self.env.width):
                if [i, j] in optimal_path:
                    optimal_grid[-1].append('X')
                else:
                    optimal_grid[-1].append('0')
        for row in optimal_grid:
            print(row)

    def get_action(self, state):
        # Returns an action using e-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            pos_actions = self.Qsa[state[0], state[1]]
            return np.random.choice(np.where(pos_actions == np.max(pos_actions))[0])

    def get_sample(self):
        # Randomly selects a previously visited state-action pair from 
        # the model dictionary
        state = random.choice(list(self.model))
        action = random.choice(list(self.model[state]))
        next_state, reward = self.model[state][action]
        return state, action, reward, next_state

    def perform_update(self, state, a, reward, new_s):
        # Updates Q-value estimates
        q = self.Qsa[state[0], state[1], a]
        delta = reward + self.gamma * np.max(self.Qsa[new_s[0], new_s[1]]) - q
        self.Qsa[state[0], state[1], a] += self.alpha * delta
    
    def solve_maze(self):
        reward_ = 0
        rewards = np.zeros(self.n_steps)    
        while self.timestep < self.n_steps:
            self.env.reset()
            state = self.env.start
            prev_steps = self.timestep
            while not self.env.done and self.timestep < self.n_steps:
                self.timestep +=1
                a = self.get_action(state)
                # get reward and new state
                reward, new_s = self.env.step(state, a)

                # Update Q-table
                self.perform_update(state, a, reward, new_s)
                # Update model
                if state not in self.model.keys():
                    self.model[state] = {}
                self.model[state][a] = [new_s, reward]
                # Planning
                for i in range(self.planning_n):
                    state, a, r, s_prime = self.get_sample()
                    self.perform_update(state, a, r, s_prime)

                if self.timestep == self.change_step:
                    self.env.change_blocked_cells()
                    self.env.reset()
                    state = self.env.start
                else:
                    state = new_s            
            # Update rewards
            rewards[prev_steps:self.timestep] = reward_
            reward_ += 1

        return rewards


class DynaQPlus(DynaQ):
    """DynaQ+ agent, inherits from DynaQ class.

    # Additional arguments
        k: float 0-1. Parameter for controlling amount of additional reward
    """

    def __init__(self, env, planning_n, n_steps, change_step,
                 alpha=0.1, epsilon=0.1, gamma=0.95,  k=1e-3):
        super().__init__(env, planning_n, n_steps, change_step,
                         alpha, epsilon, gamma)
        self.k = k
        self.timestep = 0

    def __repr__(self):
        return 'DynaQ+'

    def get_sample(self):
        """
        Randomly selects a previously visited state from model dictionary
        """
        state = random.choice(list(self.model))
        action = random.choice(list(self.model[state]))
        
        next_state, reward, tau = self.model[state][action]
        return state, action, reward, next_state, tau

    def perform_update(self, state, a, reward, new_s, additional_reward=0.0):
        # Updates Q-value estimates
        q = self.Qsa[state[0], state[1], a]
        delta = (reward + additional_reward) \
                 + self.gamma * np.max(self.Qsa[new_s[0], new_s[1]]) - q
        self.Qsa[state[0], state[1], a] += self.alpha * delta

    def solve_maze(self):
        reward_ = 0
        rewards = np.zeros(self.n_steps)
        
        while self.timestep < self.n_steps:
            self.env.reset()
            state = self.env.start
            prev_steps = self.timestep
            while not self.env.done and self.timestep < self.n_steps:
                self.timestep +=1
                a = self.get_action(state)
                # get reward and new state
                reward, new_s = self.env.step(state, a)
                # Update Q-table
                self.perform_update(state, a, reward, new_s)
                # Update model
                if state not in self.model.keys():
                    self.model[state] = {}
                    for a in range(self.n_actions):
                        self.model[state][a] = [state, 0, self.timestep]
                self.model[state][a] = [new_s, reward, self.timestep]
                # Planning
                for i in range(self.planning_n):
                    state, a, r, s_prime, tau = self.get_sample()
                    additional_reward = self.k * np.sqrt(self.timestep - tau)
                    self.perform_update(state, a, r, s_prime, additional_reward)

                if self.timestep == self.change_step:
                    self.env.change_blocked_cells()
                    self.env.reset()
                    state = self.env.start
                else:
                    state = new_s            
            # Update rewards
            rewards[prev_steps:self.timestep] = reward_
            reward_ += 1

        return rewards

class DynaQExperiment(DynaQPlus):
    """DynaQ experimental agent, inherits from DynaQPlus

    The addition reward is added at each timestep during episode,
    rather than during planning like DynaQ+
    """

    def __init__(self, env, planning_n, n_steps, change_step,
                 alpha=0.1, epsilon=0.1, gamma=0.95,  k=1e-3):
        super().__init__(env, planning_n, n_steps, change_step,
                         alpha, epsilon, gamma, k)
        self.timestep = 0

    def __repr__(self):
        return 'DynaQ Experiment'

    def solve_maze(self):
        reward_ = 0
        rewards = np.zeros(self.n_steps)
        
        while self.timestep < self.n_steps:
            self.env.reset()
            state = self.env.start
            prev_steps = self.timestep
            while not self.env.done and self.timestep < self.n_steps:
                self.timestep +=1
                a = self.get_action(state)
                # get reward and new state
                reward, new_s = self.env.step(state, a)
                # Update Q-table
                try:
                    _, _, tau = self.model[state][a]
                except KeyError:
                    # If first state-action visit set tau to current timestep
                    tau = self.timestep
                self.perform_update(state, a, reward, new_s)
                # Update model
                if state not in self.model.keys():
                    self.model[state] = {}
                    for a in range(self.n_actions):
                        self.model[state][a] = [state, 0, self.timestep]
                self.model[state][a] = [new_s, reward, self.timestep]
                # Planning
                for i in range(self.planning_n):
                    state, a, r, s_prime, tau = self.get_sample()
                    self.perform_update(state, a, r, s_prime)

                if self.timestep == self.change_step:
                    self.env.change_blocked_cells()
                    self.env.reset()
                    state = self.env.start
                else:
                    state = new_s            
            # Update rewards
            rewards[prev_steps:self.timestep] = reward_
            reward_ += 1

        return rewards


def train(env, agent_class, n_runs, planning_n, n_steps, change_step, alpha):
    results = []
    for run in range(n_runs):
        env.reset_blocked_cells()
        agent = agent_class(env, planning_n, n_steps, change_step, alpha=alpha)
        print(f'Agent: {agent}, Run {run}')
        results.append(agent.solve_maze())

    results = np.mean(results, axis=0)
    plt.plot(results, label=agent)


if __name__ == '__main__':
    n_runs = 10
    planning_n = 50
    alpha = 0.5
    agents = [DynaQ, DynaQPlus, DynaQExperiment]
    # Initialize grid as example 8.2 from the book
    blocking_maze = Environment(9, 6, (5, 3), (0, 8),
                               [(3, col) for col in range(8)],
                               [(3, col) for col in range(1, 9)])
    for agent_class in agents:
        train(blocking_maze, agent_class, n_runs, planning_n, 3000, 1000, alpha)      
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative reward')
    plt.title('Blocking maze')
    plt.legend()
    plt.savefig('images/dynaQ_ex8_2.png')
    plt.show()

    # Initialize grid as example 8.3 from the book
    shortcut_maze = Environment(9, 6, (5, 3), (0, 8),
                               [(3, col) for col in range(1, 9)],
                               [(3, col) for col in range(1, 8)])
    for agent_class in agents:
        train(shortcut_maze, agent_class, n_runs, planning_n, 6000, 3000, alpha)      
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative reward')
    plt.title('Shortcut maze')
    plt.legend()
    plt.savefig('images/dynaQ_ex8_3.png')
    plt.show()