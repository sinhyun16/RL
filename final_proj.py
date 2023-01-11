'''
gym site의 taxi-v3를 사용한다.

a. Q learinng 모델을 만든다
b. 이떄 신경망이 아닌 q-table을 사용한다.

'''

import numpy as np
import random

import gym

class QLearningAgent:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states

        self.learning_rate = 0.8
        self.discount_factor = 0.95

        self.epsilon = 0.1
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # <s, a, r, s'> 샘플로부터 큐함수 업데이트
    def learn(self, state, action, reward, next_state):
        # state, next_state = str(state), str(next_state)
        q_1 = self.q_table[state][action]
        # 벨만 최적 방정식을 사용한 큐함수의 업데이트
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

    # 큐함수에 의거하여 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            # state = str(state)
            q_list = self.q_table[state]
            action = arg_max(q_list)
        return action

    def print_qTable(self):
        print(self.q_table)

# 큐함수의 값에 따라 최적의 행동을 반환
def arg_max(q_list):
    max_idx_list = np.argwhere(q_list == np.amax(q_list))
    max_idx_list = max_idx_list.flatten().tolist()
    return random.choice(max_idx_list)


if __name__ == "__main__":
    env = gym.make('Taxi-v3')

    print(env.observation_space.n)
    print(env.action_space.n)
    print(env.render())

    agent = QLearningAgent(env.observation_space.n, env.action_space.n)

    num_episodes = 30000

    rList = []

    for i in range(num_episodes):
        current_state = env.reset()
        rewardAll = 0
        done = False
        j = 0
        while j < 99:
            j += 1

            # env.render()

            action = agent.get_action(current_state)
            next_state, reward, done, _ = env.step(action)

            agent.learn(current_state, action, reward, next_state)
            rewardAll += reward
            current_state = next_state
            if done:
                break

        rList.append(rewardAll)
        print("reward All : {}".format(rewardAll))

    print('score over time : ' + str(sum(rList) / num_episodes))
    print()
    print('final Qtable')
    agent.print_qTable()
