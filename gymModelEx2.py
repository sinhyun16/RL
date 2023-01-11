import gym
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
tf.random.set_random_seed(777)

input = tf.placeholder(shape=[1,16], dtype=tf.float32)
w = tf.Variable(tf.random_normal([16,4], stddev=0.01))
Qout = tf.matmul(input, w)
predict = tf.argmax(Qout, axis=1)

nextQ = tf.placeholder(shape=[1,4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

discount_factor = 0.99
e = 0.1
num_episodes = 2000
rList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        current_state = env.reset()
        rewardAll = 0
        done = False
        j = 0
        while j < 99:
            j+=1

            action, _Qout = sess.run([predict, Qout], feed_dict={input:np.identity(16)[current_state:current_state+1]})
            if np.random.rand(1) < e:
                action[0] = env.action_space.sample()

            next_state, reward, done, _ = env.step(action[0])
            nQ = sess.run(Qout, feed_dict={input:np.identity(16)[next_state:next_state+1]})
            maxQ = np.max(nQ)

            _Qout[0, action[0]] = reward + discount_factor * maxQ
            sess.run(train, feed_dict={input:np.identity(16)[current_state:current_state+1], nextQ:_Qout})

            rewardAll += reward
            current_state = next_state

            if done:
                e = 1./((i/50)+10)
                break

        rList.append(rewardAll)

plt.plot(rList)
plt.show()