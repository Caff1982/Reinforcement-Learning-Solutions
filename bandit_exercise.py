import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    """
    Implementation of bandit algorithm from chapter two

    k: integer. Number of arms for bandit
    epsilon (float) controls exploration/exploitation trade-off
    """
    def __init__(self, epsilon, k=10, mean=0, std=1,
                 stepsize=0.1, sample_avg=False):
        self.epsilon = epsilon
        self.k = k
        self.stepsize = stepsize
        self.sample_avg = sample_avg

        self.mean = mean
        self.std = std
        self.Q_true = np.random.normal(self.mean, self.std, self.k)
        self.best_action = np.argmax(self.Q_true)

        self.Q_estimates = np.zeros(k)
        self.N_actions = np.zeros(k)

    def __str__(self):
        """
        Returns description of Bandit for plotting purposes
        """
        return f'Agent with epsilon: {self.epsilon}'

    def reset(self):
        self.Q_true = np.random.normal(self.mean, self.std, self.k)
        self.best_action = np.argmax(self.Q_true)

        self.Q_estimates = np.zeros(self.k)
        self.N_actions = np.zeros(self.k)

    def get_action(self):
        """
        Chooses action using e-greedy policy
        """
        if np.random.random() < self.epsilon:
            # Exploration
            action = np.random.randint(self.k)

        else:
            # Exploitation
            action = np.argmax(self.Q_estimates)
        return action

    def update(self, action):
        self.N_actions[action] += 1

        reward = np.random.normal(self.Q_true[action])
        if self.sample_avg:
            self.Q_estimates[action] += (reward - self.Q_estimates[action]) / self.N_actions[action]
        else: # Use constant stepsize
            self.Q_estimates[action] += self.stepsize * (reward - self.Q_estimates[action])
        
        # Update true Q values
        self.Q_true += np.random.normal(self.mean, 0.01, self.k)

        return reward


def train(bandits, n_iters, timesteps):
    rewards = np.zeros((len(bandits), n_iters, timesteps))
    best_actions = np.zeros((len(bandits), n_iters, timesteps))

    for i, bandit in enumerate(bandits):
        for iteration in range(n_iters):
            bandit.reset()
            if not iteration % 500:
                print('Iteration:', iteration)
            for step in range(timesteps):
                action = bandit.get_action()
                reward = bandit.update(action)
                # Update rewards
                rewards[i, iteration, step] = reward
                if action == bandit.best_action:
                    best_actions[i, iteration, step] = 1

    mean_rewards = rewards.mean(axis=1)
    mean_best_actions = best_actions.mean(axis=1)
    return mean_rewards, mean_best_actions


if __name__ == '__main__':
    num_arms = 10
    mean = 0
    std = 1
    n_iters = 2000
    timesteps = 1000

    # Constant stepsize, 10k steps
    bandits = [Bandit(epsilon=0.1), Bandit(epsilon=0.1, sample_avg=True)]
    rewards, best_actions = train(bandits, n_iters, 10000)

    plt.plot(rewards[0], label='Bandit using stepsize 0.1')
    plt.plot(rewards[1], label='Bandit using sample averages')
    plt.xlabel('Timesteps')
    plt.ylabel('Average Return')
    plt.legend()
    plt.savefig('images/ex2_stepsize_10k')
    plt.show()

    # bandit = Bandit(epsilon=0.1, sample_avg=True)
    # print(bandit.Q_true)