import numpy as np
import matplotlib.pyplot as plt


class PolicyIterationSolver:
    """Class for policy evaluation and improvement
    state is represented by tuple of (num cars at location a, num cars at
    location b)
    """

    def __init__(self, max_cars=20, max_transfer=5, reward=10,
                 transfer_cost=2, gamma=0.9, min_delta=1e-3, 
                 exp_requests_a=3, exp_requests_b=4, exp_returns_a=3,
                 exp_returns_b=2, poisson_upper_bound=11, solve_exercise=False):
        self.max_cars = max_cars
        self.max_transfer = max_transfer
        self.reward = reward
        self.transfer_cost = transfer_cost
        self.gamma = gamma
        self.min_delta = min_delta

        self.exp_requests_a = exp_requests_a
        self.exp_requests_b = exp_requests_b
        self.exp_returns_a = exp_returns_a
        self.exp_returns_b = exp_returns_b
        # Set upper bound for poisson distribution
        self.poisson_upper_bound = poisson_upper_bound
        # If True solves exercise 4.5, else creates figure 4.2
        self.solve_exercise = solve_exercise

        self.policy = np.zeros((self.max_cars + 1, self.max_cars + 1))
        self.values = np.zeros((self.max_cars + 1, self.max_cars + 1))
        self.actions = np.arange(-self.max_transfer, self.max_transfer+1)

        self.poisson_dict = self.create_poisson_dict()

    def create_poisson_dict(self):
        """
        Returns a dictionary where the keys are tuple of (num cars requested,
        expected requests) and values are the poisson probability
        """
        poisson_dict = {}
        for n_req in range(self.poisson_upper_bound):
            for exp_req in (self.exp_requests_a, self.exp_requests_b):
                poisson_dict[(n_req, exp_req)] = ((exp_req**n_req) \
                                    / np.math.factorial(n_req)) * np.exp(-exp_req)
        return poisson_dict

    def get_expected_return(self, state, action):
        # Initialize expected reward as zero minus the cost of transferring cars
        returns = 0.0
        if self.solve_exercise and action > 0:
            # Free shuttle service
            returns = -self.transfer_cost * np.abs(action-1)
        else:
            returns = -self.transfer_cost * np.abs(action)

        # Update number of cars at each location
        cars_a = min(state[0] - action, self.max_cars)
        cars_b = min(state[1] + action, self.max_cars)

        if self.solve_exercise:
            if cars_a > 10:
                returns -= 4
            if cars_b > 10:
                returns -= 4

        # Loop through rental requests
        for req_a in range(self.poisson_upper_bound):
            for req_b in range(self.poisson_upper_bound):
                # get rental probabilities
                rental_proba = self.poisson_dict[req_a, self.exp_requests_a] \
                               * self.poisson_dict[req_b, self.exp_requests_b]
                # Get num of cars at each location for current state
                n_cars_a = cars_a
                n_cars_b = cars_b
                # Get the valid number of rentals
                valid_rentals_a = min(n_cars_a, req_a)
                valid_rentals_b = min(n_cars_b, req_b)
                # Get total rewards for rentals at both locations
                rewards = (valid_rentals_a + valid_rentals_b) * self.reward
                n_cars_a -= valid_rentals_a
                n_cars_b -= valid_rentals_b

                # Returned cars
                n_cars_a = int(min(n_cars_a + self.exp_returns_a, self.max_cars))
                n_cars_b = int(min(n_cars_b + self.exp_returns_b, self.max_cars))
                returns += rental_proba * (rewards + self.gamma \
                                        * self.values[(n_cars_a, n_cars_b)])
        return returns

    def policy_evaluation(self):
        print('Policy evaluation')
        while True:
            old_values = self.values.copy()
            for a in range(self.max_cars+1):
                for b in range(self.max_cars+1):
                    state = (a, b)
                    action = self.policy[state]
                    self.values[state] = self.get_expected_return(state, action)
            
            max_value_change = np.abs(old_values - self.values).max()
            print('Max value change:', max_value_change)
            if max_value_change < self.min_delta:
                break

    def policy_improvement(self):
        print('Policy improvement')
        for a in range(self.max_cars+1):
            for b in range(self.max_cars+1):
                state = (a, b)
                old_action = self.policy[state]

                action_rewards = []
                for action in self.actions:
                    # Check action is valid
                    if (0 <= action <= state[0]) or (-state[1] <= action <= 0):
                        action_rewards.append(self.get_expected_return(state, action))
                    else:
                        action_rewards.append(-float('inf'))
                
                opt_action = np.argmax(action_rewards)
                self.policy[state] = self.actions[opt_action]

    def plot_policy(self, step, title):
        plt.imshow(self.policy[::-1], cmap='viridis')
        plt.ylabel('Cars at location A')
        plt.xlabel('Cars at location B')
        yticks = range(self.max_cars+1)
        plt.yticks(yticks, yticks[::-1])
        plt.colorbar()
        plt.title(f'Policy after step {step+1}')
        plt.savefig(f'images/{title}_step{step+1}.png')
        plt.close()


if __name__ == '__main__':
    # Recreating figure 4.2
    solver = PolicyIterationSolver()
    for step in range(4):
        solver.policy_evaluation()
        solver.policy_improvement()
        solver.plot_policy(step, 'fig4_2')

    # Solving exercise 4.7
    solver = PolicyIterationSolver(solve_exercise=True)
    for step in range(4):
        solver.policy_evaluation()
        solver.policy_improvement()
        solver.plot_policy(step, 'ex4_7')
