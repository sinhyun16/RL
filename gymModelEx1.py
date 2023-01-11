import gym
import numpy as np

env = gym.make('FrozenLake-v0')
print(env.observation_space.n) #16
print(env.action_space.n) #4
print(env.render())

learning_rate = 0.8
discount_factor = 0.95
num_episodes = 2000
Q_table = np.zeros([env.observation_space.n, env.action_space.n]) #16 x 4

rList = []
# current_state = env.reset()
# print(current_state)
for i in range(num_episodes):
    current_state = env.reset()
    rewardAll = 0
    done = False
    j = 0
    while j < 99:
        j += 1
        action = np.argmax(Q_table[current_state, :] + np.random.randn(1, env.action_space.n)*(1./(i+1)))

        next_state, reward, done, _ = env.step(action)

        Q_table[current_state, action] = Q_table[current_state, action] + \
                                         learning_rate * (reward + discount_factor * np.max(Q_table[next_state, :]) -
                                                          Q_table[current_state, action])
        rewardAll += reward
        current_state = next_state
        if done:
            break
    rList.append(rewardAll)

print('score over time : ' + str(sum(rList) / num_episodes))
print()
print('final Qtable')
print(Q_table)