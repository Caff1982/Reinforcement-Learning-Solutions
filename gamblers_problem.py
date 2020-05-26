import numpy as np
import matplotlib.pyplot as plt


def value_iteration(heads_proba, theta=1e-10, max_capital=100):
    values = np.zeros(max_capital+1)
    values[max_capital] = 1
    values[0] = 0

    counter = 0
    while True:
        old_state_value = values.copy()
        for state in range(1, len(values) - 1):
            temp_values = []
            for action in range(min(state, max_capital -state) + 1):
                exp_return = heads_proba * values[state + action] \
                             + (1-heads_proba) * values[state - action]
                temp_values.append(exp_return)
            
            opt_action = np.argmax(temp_values)
            values[state] = temp_values[opt_action]

        counter += 1
        if not counter % 50:
            print(f'Step: {counter}, Delta: {delta}')

        delta = abs(old_state_value - values).max()
        if delta < theta:
            break
    return values

def greedy_policy(values, heads_proba, max_capital=100):
    policy = np.zeros(max_capital + 1)
    for state in range(1, len(values) - 1):
        temp_values = []
        for action in range(min(state, max_capital -state) + 1):
            exp_return = heads_proba * values[state + action] \
                         + (1-heads_proba) * values[state - action]
            temp_values.append(exp_return)
        policy[state] = np.argmax(temp_values)
    return policy

def plot_results(policy, proba):
    plt.plot(policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy')
    plt.title(f'Final policy for heads_proba {proba}')
    plt.xticks(range(0, 100, 5))
    plt.savefig(f'images/ex4_9_proba_{proba}.png')
    plt.show()


if __name__ == '__main__':
    probabilities = [0.4, 0.25, 0.55]
    for proba in probabilities:
        values = value_iteration(proba)
        policy = greedy_policy(values, proba)
        plot_results(policy, proba)
