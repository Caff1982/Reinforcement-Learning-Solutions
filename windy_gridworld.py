import numpy as np
import matplotlib.pyplot as plt


class Environment:

    def __init__(self, start, goal, height, width, use_stochastic=False):
        self.start = start
        self.goal = goal
        self.height = height
        self.width = width
        self.use_stochastic = use_stochastic

        self.winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        # Array mapping actions to moves
        self.action2move = np.array([[-1, 0], [1, 0], [0, -1], [0, 1],
                                     [-1, -1],[-1, 1],[1, -1],[1, 1],[0,0]])
        self.done = False

    def reset(self):
        self.done = False

    def update(self, state, action):
        y, x = state
        y_delta, x_delta = self.action2move[action]

        if self.use_stochastic:
            rand_proba = np.random.random()
            if rand_proba < 1/3:
                state = [y + y_delta, x + x_delta]
            elif rand_proba > 2/3:
                state = [y + y_delta - self.winds[x], x + x_delta]
            else:
                state = [y + y_delta - 2 * self.winds[x], x + x_delta]
        else:
            state = [y + y_delta - self.winds[x], x + x_delta]

        state[0] = np.clip(state[0], 0, self.height-1)
        state[1] = np.clip(state[1], 0, self.width-1)
        if state == self.goal:
            self.done = True
            return 0, state
        else:
            return -1, state


class Agent:

    def __init__(self, env, start, height, width, gamma=1,
                 alpha=0.5, epsilon=0.1, n_actions=4):
        self.env = env
        self.start = start
        self.height = height
        self.width = width
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_actions = n_actions
        # Q-table to store state-action estimates
        self.Qsa = np.zeros((self.height, self.width, self.n_actions))

    def __repr__(self):
        """
        Returns string with number of actions, used for plotting
        """
        return f'Agent with {self.n_actions} actions.'

    def get_action(self, state):
        """
        Returns action chosen using e-greedy policy
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            actions = self.Qsa[state[0], state[1]]
            return np.random.choice(np.where(actions == np.max(actions))[0])

    def play_episode(self):
        """
        On-policy SARSA control algorithm.
        Returns number of steps in episode as an integer
        """
        state = self.start
        action = self.get_action(state)
        n_steps = 0
        while not self.env.done:
            # Get reward and next state
            r, new_state = self.env.update(state, action)
            new_action = self.get_action(new_state)
            # Update Q-estimates
            self.Qsa[state[0], state[1], action] += self.alpha * (r + self.gamma * \
                            self.Qsa[new_state[0], new_state[1], new_action]\
                            - self.Qsa[state[0], state[1], action])
            n_steps += 1
            state = new_state
            action = new_action

        return n_steps

def get_optimal_path(env, agent, start=[3,0], goal=[3,7]):
    """
    Takes a Q-table of state-action pairs as input.
    Returns optimal path as a list
    """
    optimal_path = []
    state = start
    while state != goal:
        optimal_path.append(state)
        action = np.argmax(agent.Qsa[state[0], state[1]])
        r, new_state = env.update(state, action)

        state = new_state
    return optimal_path

def train(env, agent, n_episodes):
    episode_lengths = []
    for i in range(n_episodes):
        env.reset()
        steps = agent.play_episode()
        episode_lengths.append(steps)
    return episode_lengths


if __name__ == '__main__':
    height = 7
    width = 10
    start = [3, 0]
    goal = [3, 7]
    env = Environment(start, goal, height, width)
    agents = [Agent(env, start, height, width, n_actions=4),
              Agent(env, start, height, width, n_actions=8),
              Agent(env, start, height, width, n_actions=9)]

    # Exercise 6.9, Windy Gridworld with King's Moves
    for agent in agents:
        train_history = train(env, agent, 170)
        train_history = np.add.accumulate(train_history)

        plt.plot(train_history, np.arange(1, len(train_history) + 1),
                 label=agent)
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.title('Windy Gridworld')
    plt.legend()
    plt.savefig('images/ex6_9.png')
    plt.show()

    # Exercise 6.10, Stochastic winds
    env = Environment(start, goal, height, width,
                      use_stochastic=True)
    agents = [Agent(env, start, height, width, n_actions=4),
              Agent(env, start, height, width, n_actions=8),
              Agent(env, start, height, width, n_actions=9)]
    
    for agent in agents:
        train_history = train(env, agent, 400)
        train_history = np.add.accumulate(train_history)
        plt.plot(train_history, np.arange(1, len(train_history) + 1),
                 label=agent)

    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.title('Stochastic Gridworld')
    plt.legend()
    plt.savefig('images/ex6_10.png')
    plt.show()


