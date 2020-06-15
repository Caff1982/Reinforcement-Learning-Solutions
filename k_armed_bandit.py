import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    """K-armed bandit from chapter 2

    Arguments:
        epsilon (float 0-1): Controls e-greedy exploration/exploitation
        k (int): Number of arms on bandit
        non_stationary (bool): If True adds 0.01 noise at each step
        alpha (float 0-1): Stepsize parameter
        mean: sets the mean of true values
        std: sets standard deviation of true values

    Qa: The Q-estimate for each action
    Na: The number of times an action has been performed
    """

    def __init__(self, epsilon, k=10, mean=0, std=1,
                 non_stationary=False, alpha=None):
        self.epsilon = epsilon
        self.k = k
        self.mean = mean
        self.std = std
        self.non_stationary = non_stationary
        self.alpha = alpha

        self.reset()

    def __str__(self):
        """
        Returns description of Bandit for plotting
        """
        if self.alpha:
            return f'Agent with alpha: {self.alpha}, epsilon: {self.epsilon}'
        else:
            return f'Agent with epsilon: {self.epsilon}'
        

    def reset(self):
        self.true_values = np.random.normal(self.mean, self.std, self.k)
        self.opt_action = np.argmax(self.true_values)
        self.Qa = np.zeros(self.k)
        self.Na = np.zeros(self.k)

    def get_action(self):
        """
        Returns an action using e-greedy policy
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.Qa)

    def update(self, action):
        self.Na[action] += 1
        reward = np.random.normal(self.true_values[action])
        
        if self.alpha: # Use alpha as stepsize param
            self.Qa[action] += self.alpha * (reward - self.Qa[action])
        else: # Sample average
            self.Qa[action] += (reward - self.Qa[action]) / self.Na[action]
        # If non_stationary add noise
        if self.non_stationary:
            self.Qa += np.random.normal(self.mean, 0.01, self.k)

        return reward


def train(bandits, n_iters, n_steps):
    """Trains bandits and returns mean rewards and optimal actions

    Arguments:
        bandits (list): A list of Bandits to be trained
        n_iters (int): The number of iterations to average over
        n_steps (int). The number of steps per iteration

    Returns:
        The mean results and percentage of optimal actions taken
        for each bandit, averaged over n_iters
    """
    rewards = np.zeros((len(bandits), n_iters, n_steps))
    opt_actions = np.zeros((len(bandits), n_iters, n_steps))
    for i, bandit in enumerate(bandits):
        print(f'Training: {bandit}')
        for iteration in range(n_iters):
            if not iteration % 500:
                print(f'Iteration: {iteration}')
            bandit.reset()
            for step in range(n_steps):
                action = bandit.get_action()
                reward = bandit.update(action)
                rewards[i, iteration, step] = reward
                if action == bandit.opt_action: # increment optimal action
                    opt_actions[i, iteration, step] = 1

    mean_rewards = rewards.mean(axis=1)
    mean_opt_actions = opt_actions.mean(axis=1) * 100

    return mean_rewards, mean_opt_actions

def plot_results(bandits, rewards, opt_actions, title):
    for i, bandit in enumerate(bandits):
        plt.plot(rewards[i], label=bandit)
    plt.xlabel('Timesteps')
    plt.ylabel('Average reward')
    plt.legend()
    plt.savefig(f'images/{title}_rewards.png')
    plt.close()

    for i, bandit in enumerate(bandits):
        plt.plot(opt_actions[i], label=bandit)
    plt.xlabel('Timesteps')
    plt.ylabel('% Optimal action')
    plt.legend()
    plt.savefig(f'images/{title}_actions.png')
    plt.close()


if __name__ == '__main__':
    # Recreating figure 2.2
    bandits = [Bandit(0.1), Bandit(0.01), Bandit(0.0)]
    rewards, opt_actions = train(bandits, 3000, 1000)
    plot_results(bandits, rewards, opt_actions, 'fig2_2')

    # Exercise 2.5
    bandits = [Bandit(0.1, non_stationary=True),
               Bandit(0.1, non_stationary=True, alpha=0.1)]
    rewards, opt_actions = train(bandits, 2000, 10000)
    plot_results(bandits, rewards, opt_actions, 'ex2_5')
