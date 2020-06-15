import numpy as np
import matplotlib.pyplot as plt


class Environment:

    def __init__(self):
        self.height = 4
        self.width = 12
        self.start = [3, 0]
        self.goal = [3, 11]

        self.action2move = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        self.done = False

    def reset(self):
        self.done = False

    def update(self, state, action):
        i, j = state
        i_delta, j_delta = self.action2move[action]

        state = [i + i_delta, j + j_delta]

        state[0] = np.clip(state[0], 0, self.height-1)
        state[1] = np.clip(state[1], 0, self.width-1)

        if state == self.goal:
            self.done = True
            return 0, state
        elif state[0] == 3 and state[1] in range(1, 11):
            return -100, self.start
        else:
            return -1, state

class Agent:

    def __init__(self, env, use_SARSA=True):
        self.env = env
        self.use_SARSA = use_SARSA
        self.n_actions = 4
        self.epsilon = 0.1
        self.gamma = 1.0
        self.alpha = 0.5

        self.Qsa = np.zeros((self.env.height, self.env.width, self.n_actions))

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
        s = self.env.start
        rewards = []
        while not self.env.done:
            a = self.get_action(s)
            # Get reward and next state
            r, new_state = self.env.update(s, a)
            new_action = self.get_action(new_state)

            if self.use_SARSA:
                self.Qsa[s[0], s[1], a] += self.alpha * (r + self.gamma * \
                                self.Qsa[new_state[0], new_state[1], new_action]\
                                - self.Qsa[s[0], s[1], a])
            else: # Q-learning
                self.Qsa[s[0], s[1], a] += self.alpha * (r + self.gamma * \
                                np.max(self.Qsa[new_state[0], new_state[1]])\
                                - self.Qsa[s[0], s[1], a])
            s = new_state
            a = new_action
            rewards.append(r)
        return np.sum(rewards)

    def print_optimal_path(self):
        optimal_path = []
        state = self.env.start

        while True:
            optimal_path.append(state)
            action = np.argmax(self.Qsa[state[0], state[1]])
            _, state = self.env.update(state, action)

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

def train(env, agent, n_episodes):
    rewards = []
    for i in range(n_episodes):
        env.reset()
        rewards.append(agent.play_episode())

    return agent, rewards


if __name__ == '__main__':
    env = Environment()
    agent = Agent(env)
    n_episodes = 500

    sarsa_rewards = []
    q_rewards = []
    for i in range(1000):
        if not i % 20:
            print(f'Training loop: {i}')
        sarsa_agent, sarsa_r = train(env, Agent(env), n_episodes)
        q_agent, q_r = train(env, Agent(env, use_SARSA=False), n_episodes)
        sarsa_rewards.append(sarsa_r)
        q_rewards.append(q_r)

    sarsa_rewards = np.mean(sarsa_rewards, axis=0)
    q_rewards = np.mean(q_rewards, axis=0)

    plt.plot(sarsa_rewards, label='Sarsa')
    plt.plot(q_rewards, label='Q-learning')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.ylim(bottom=-100, top=-15)
    plt.legend()
    plt.savefig('images/cliffwalk.png')
    plt.show()

    print('Optimal path SARSA')
    sarsa_agent.print_optimal_path()
    print('Optimal path Q-Learning')
    q_agent.print_optimal_path()