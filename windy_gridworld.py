import numpy as np
import matplotlib.pyplot as plt


class Environment:
    """
    Class for representing windy gridworld environment.
    """

    def __init__(self, start, goal, height, width, use_stochastic=False):
        self.start = start
        self.goal = goal
        self.height = height
        self.width = width
        self.use_stochastic = use_stochastic
        
        self.winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0] # wind for each colum
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


def train(env, n_runs, n_episodes, agent_actions):
    """
    Completes training for n_episodes over n_runs for the agents specified 
    in agent_actions (a list of ints).
    Returns a 2D list of lists of mean episode-lengths for each agent
    """
    agents = [Agent(env, start, height, width, n_actions=n) for n in agent_actions]
    agent_rewards = [[] for agent in range(len(agents))]
    for run in range(n_runs):
        if run % 10 == 0:
            print(f'Run: {run}')
        for i, agent in enumerate(agents):
            episode_lengths = []
            for ep in range(n_episodes):
                env.reset()
                steps = agent.play_episode()
                episode_lengths.append(steps)
            agent_rewards[i].append(np.add.accumulate(episode_lengths))
    return np.mean(agent_rewards, axis=1)

def plot(agent_rewards, agent_actions, title, filename):
    for i in range(len(agent_rewards)):
        plt.plot(agent_rewards[i], np.arange(1, len(agent_rewards[i]) + 1),
                 label=f'Agent with {agent_actions[i]} actions')
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.title(title)
    plt.legend()
    plt.savefig(f'images/{filename}.png')
    plt.show()

if __name__ == '__main__':
    height = 7
    width = 10
    start = [3, 0]
    goal = [3, 7]
    agent_actions = [4, 8, 9]
    env = Environment(start, goal, height, width)
    
    # Exercise 6.9, Windy Gridworld with King's Moves
    train_history = train(env, 20, 170, agent_actions)
    plot(train_history, agent_actions, 'Windy Gridworld', 'ex6_9')

    # Exercise 6.10, Stochastic winds
    env = Environment(start, goal, height, width, use_stochastic=True)
    train_history = train(env, 50, 500, agent_actions)
    plot(train_history, agent_actions, 'Stochastic Wind', 'ex6_10')
