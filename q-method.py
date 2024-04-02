import numpy as np

# Определение среды
num_states = 6
num_actions = 2
q_table = np.zeros((num_states, num_actions))

# Функция выбора действия с epsilon-greedy
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(q_table[state, :])

# Параметры Q-обучения
learning_rate = 0.1
discount_factor = 0.9
num_episodes = 1000
epsilon = 0.2

# Обучение агента
for episode in range(num_episodes):
    state = 0
    done = False

    while not done:
        action = select_action(state, epsilon)
        next_state = state + action
        reward = 0 if next_state != 5 else 1

        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]))

        state = next_state

        if state == 5:
            done = True

print("Q-таблица:")
print(q_table)