import numpy as np

from racetracks import test_track, track1
from gui import GUI

"""
On policy implementation. Works but rewards are 1 for goal and 0 elsewhere

Also velocity is between -4 and 4, should be bounded.
And zero velocity should not be allowed
"""

class Environment:
    """Class to represent the ractrack as 2D array"""

    def __init__(self, track):
        self.track = track
        self.done = False

        self.start_cells = np.where(self.track[-1:] == 2)[1]
        self.action_moves = np.array([[0, 0], [0, 1], [0, -1],
                                      [-1, 0], [-1, 1], [-1, -1],
                                      [1, 0], [1, 1], [1, -1]])
        self.velocity = np.array([0, 0])

    def reset(self):
        self.pos = np.array([len(self.track)-1, np.random.choice(self.start_cells)])
        self.velocity = np.array([0, 0])

        self.done = False

    def get_state(self):    
        return (self.pos[0], self.pos[1], self.velocity[0], self.velocity[1])

    def is_valid_cell(self):
        if 0 <= self.pos[0] < len(self.track) and 0 <= self.pos[1] < len(self.track[0]):         
            if self.track[self.pos[0], self.pos[1]] in (0, 1, 2):
                return True
        self.done = True
        return False

    def reached_end(self):
        if self.pos[0] >= 0 and self.pos[1] >  len(self.track[0]):
            print('Reached end')
            # print('Position:', self.pos)
            self.done = True
            return True
        return False

    def set_velocity(self, action):
        self.velocity += self.action_moves[action]
        self.velocity = np.clip(self.velocity, -4, 4)

        # self.velocity[0] = np.clip(self.velocity[0], -4, 0)
        # self.velocity[1] = np.clip(self.velocity[1], 0, 4)
        # if (self.velocity == np.array([0, 0])).all():
        #     self.velocity[0] = -np.random.randint(2)
        #     self.velocity[0] = np.random.randint(2)

    def update(self, action):
        self.velocity += self.action_moves[action]
        self.velocity = np.clip(self.velocity, -4, 4)
        self.pos += self.velocity

        if self.reached_end():
            self.done = True
            return 1
        elif self.is_valid_cell():
            return 0
        else:
            self.done = True
            return 0

class MonteCarloSim:
    """Class to simulates episodes using Monte-Carlo"""
    def __init__(self, env, add_noise=False, epsilon=0.1,
                 gamma=0.9, n_speeds=5, n_actions=9):
        self.env = env
        # add_noise creates 0.1 proba of velocity being zero
        self.add_noise = add_noise
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
            # if self.add_noise:
            #     if np.random.random() < 0.1:
            #         action = [0, 0]
            reward = self.env.update(action)

            self.sequence.append((state, action, reward))

    def policy_evaluation(self,):
        """On-policy incremental implemenatation"""
        returns = np.zeros(len(self.sequence))
        G = 0
        for i in reversed(range(len(self.sequence))):
            state, action, reward = self.sequence[i]
            state_action = (*state, action)

            G = self.gamma * G + reward
            self.action_counts[state_action] += 1
            self.Qsa[state_action] += update_mean(G, self.Qsa[state_action],
                                                  self.action_counts[state_action])

            # self.policy[state] = np.argmax(self.Qsa[state])

    def update_policy(self):
        # Greedy update policy, choose move with max expected return
        self.policy = np.argmax(self.Qsa, axis=-1)


def update_mean(reward, old_value, action_count):
    # Calculates amount to add for running average
    return (reward - old_value) / (action_count + 1)



if __name__ == '__main__':
    env = Environment(track1)
    mc = MonteCarloSim(env)

    for ep in range(100000):
        env.reset()
        mc.play_episode()
        mc.policy_evaluation()
        
        mc.update_policy()

        if not ep % 5000:
            print(f'Episode: {ep}, Sequence length: {len(mc.sequence)}')

    gui = GUI(track1)
    print(mc.sequence)
    gui.plot_sequence(mc.sequence)