'''
gym 사이트의 mountainCar-v0를 사용한다.
a. 신경망을 이용하여 강화학습 모델을 만든다
b. 모델을 학습 후 10개의 에피소드를 시현하는 코드를 구성한다.
'''

import os
import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform


# 상태가 입력, 큐함수가 출력인 인공신경망 생성
class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q

# 카트폴 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = True

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.1
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.01

        self.model = DQN(action_size)
        self.optimizer = Adam(lr=self.learning_rate)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            return np.argmax(q_value[0])

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predict = self.model(state)[0][action]

            next_q = self.model(next_state)[0][next_action]
            target = reward + (1 - done) * self.discount_factor * next_q

            loss = tf.reduce_mean(tf.square(target-predict))

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    arriveTime = 0
    num_episode = 1000
    for e in range(num_episode):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)

            position_max = 0.5

            if reward < 1:
                if next_state[0][0] > -0.3 and action == 2:
                    reward = ((next_state[0][0] + 0.3) / position_max) * 10
                elif next_state[0][0] < 0.5 and action == 1:
                    reward = -1
                else:
                    reward = 0

            agent.train_model(state, action, reward, next_state, next_action, done)

            score += reward
            state = next_state

            if done:
                print("episode: {:3d} | score : {:.4f} | epsilon: {:.4f}".format(
                      e, score, agent.epsilon))

                # 에피소드마다 학습 결과 그래프로 저장
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./graph.png")

                if next_state[0][0] >= 0.5:
                    print("*"*100)
                    print("{}/{}arrive!".format(arriveTime+1, 100))
                    print("*" * 100)
                    arriveTime += 1

                if arriveTime == 100:
                    agent.model.save_weights("./model", save_format="tf")
                    sys.exit()
