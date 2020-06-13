import numpy as np

from racetracks import test_track, track1
from gui import GUI


class Environment:
    """
    Class to represent ractrack as 2D array
    """

    def __init__(self, track, add_noise=True):
        self.track = track
        # add_noise creates 0.1 proba of velocity being zero
        self.add_noise = add_noise

        self.start_cells = np.where(self.track[-1:] == 2)[1]
        self.action_moves = np.array([[0, 0], [0, 1], [0, -1],
                                      [-1, 0], [-1, 1], [-1, -1],
                                      [1, 0], [1, 1], [1, -1]])
        self.reset()

    def reset(self):
        """
        Resets to inital settings, start position chosen randomly
        """
        self.pos = np.array([len(self.track)-1, np.random.choice(self.start_cells)])
        self.velocity = np.array([0, 0])
        self.done = False

    def get_state(self):    
        return (self.pos[0], self.pos[1], self.velocity[0], self.velocity[1])

    def is_valid_cell(self):
        if 0 <= self.pos[0] < len(self.track) and 0 <= self.pos[1] < len(self.track[0]):         
            if self.track[self.pos[0], self.pos[1]] != 3:
                return True
        self.done = True
        return False

    def reached_end(self):
        if self.pos[0] >= 0 and self.pos[1] >  len(self.track[0]):
            self.done = True
            return True
        return False

    def update(self, action):
        # Update velocity and clip values
        self.velocity += self.action_moves[action]
        self.velocity[0] = np.clip(self.velocity[0], -4, 0)
        self.velocity[1] = np.clip(self.velocity[1], 0, 4)
        # If velocity is zero select up or right randomly
        if (self.velocity == np.array([0, 0])).all():
            if np.random.random() > 0.5:
                self.velocity = np.array([-1, 0])
            else:
                self.velocity = np.array([0, 1])

        if self.add_noise:
            if np.random.random() < 0.1:
                velocity = np.array([0, 0])

        self.pos += self.velocity
        if self.reached_end():
            self.done = True
            return 0
        elif self.is_valid_cell():
            return -1
        else:
            # apply out-of-bounds penalty
            self.done = True
            return -20

class MonteCarloSim:
    """
    Class to simulate episodes using Monte-Carlo
    """

    def __init__(self, env, epsilon=0.1, gamma=0.9,
                 n_speeds=5, n_actions=9):
        self.env = env

        self.epsilon = 0.1
        self.gamma = 0.9
        self.n_speeds = 5
        self.n_actions = 9       
        self.sequence = []

        self.Qsa = np.zeros((len(self.env.track), len(self.env.track[0]), 
                             self.n_speeds, self.n_speeds, self.n_actions))
        self.action_counts = np.zeros((len(self.env.track), len(self.env.track[0]),
                                       self.n_speeds, self.n_speeds, self.n_actions))
        self.policy = np.zeros((len(self.env.track), len(self.env.track[0]),
                                self.n_speeds, self.n_speeds), dtype=np.int16)

    def get_action(self, state):
        # Returns an action using e-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return self.policy[state]

    def play_episode(self):
        self.sequence = []
        while not self.env.done:
            state = self.env.get_state()
            action = self.get_action(state)
            reward = self.env.update(action)
            
            self.sequence.append((state, action, reward))

    def policy_evaluation(self,):
        """
        On-policy incremental implemenatation using importance sampling
        """
        returns = np.zeros(len(self.sequence))
        G = 0
        for i in reversed(range(len(self.sequence))):
            state, action, reward = self.sequence[i]
            state_action = (*state, action)
            G = self.gamma * G + reward

            self.action_counts[state_action] += 1
            self.Qsa[state_action] += (1 / self.action_counts[state_action]) * \
                                      (G - self.Qsa[state_action])

    def policy_improvement(self):
        """
        Greedy policy update, selects actions with max expected return
        """
        self.policy = np.argmax(self.Qsa, axis=-1)


def get_optimal_path(policy):
    env = Environment(track1, add_noise=False)
    optimal_path = []
    while not env.done:
        state = env.get_state()
        optimal_path.append(state)

        action = policy[state]
        env.update(action)
    return optimal_path

def update_mean(reward, old_value, action_count):
    """
    Calculates amount to add for running average
    """
    return (reward - old_value) / (action_count + 1)


if __name__ == '__main__':
    env = Environment(track1)
    mc = MonteCarloSim(env)

    for episode in range(100000):
        env.reset()
        mc.play_episode()
        mc.policy_evaluation()
        mc.policy_improvement()

        if not episode % 5000:
            print(f'Episode: {episode}, Sequence length: {len(mc.sequence)}')

    optimal_path = get_optimal_path(mc.policy)
    print(optimal_path)
    gui = GUI(track1)
    gui.plot_sequence(optimal_path)