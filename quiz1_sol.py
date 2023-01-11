import numpy as np
import time
import copy

def show_v_table(v_table, env):
    for i in range(env.reward.shape[0]):
        print("+-----------------"*env.reward.shape[1],end="")
        print("+")
        for k in range(3):
            print("|",end="")
            for j in range(env.reward.shape[1]):
                if k==0:
                    print("                 |",end="")
                if k==1:
                        print("   {0:8.2f}      |".format(v_table[i,j]),end="")
                if k==2:
                    print("                 |",end="")
            print()
    print("+-----------------"*env.reward.shape[1],end="")
    print("+")


def show_policy(policy,env):
    for i in range(env.reward.shape[0]):
        print("+-----------------"*env.reward.shape[1],end="")
        print("+")
        for k in range(3):
            print("|",end="")
            for j in range(env.reward.shape[1]):
                if k==0:
                    print("                 |",end="")
                if k==1:
                    if policy[i,j] == 0:
                        print("      ↑         |",end="")
                    elif policy[i,j] == 1:
                        print("      →         |",end="")
                    elif policy[i,j] == 2:
                        print("      ↓         |",end="")
                    elif policy[i,j] == 3:
                        print("      ←         |",end="")
                if k==2:
                    print("                 |",end="")
            print()
    print("+-----------------"*env.reward.shape[1],end="")
    print("+")


class Agent():

    action = np.array([[-1,0],[0,1],[1,0],[0,-1]])
    select_action_pr = np.array([0.25,0.25,0.25,0.25])

    def __init__(self):
        self.pos = (0,0)

    def set_pos(self,position):
        self.pos = position
        return self.pos

    def get_pos(self):
        return self.pos


class Environment():

    cliff = -3
    road = -1
    goal = 1

    goal_position = [2,2]

    reward_list = [[road,road,road],
                   [road,road,road],
                   [road,road,goal]]

    reward_list1 = [["road","road","road"],
                    ["road","road","road"],
                    ["road","road","goal"]]

    def __init__(self):
        self.reward = np.asarray(self.reward_list)

    def move(self, agent, action):

        done = False
        new_pos = agent.pos + agent.action[action]

        if self.reward_list1[agent.pos[0]][agent.pos[1]] == "goal":
            reward = self.goal
            observation = agent.set_pos(agent.pos)
            done = True
        elif new_pos[0] < 0 or new_pos[0] >= self.reward.shape[0] or new_pos[1] < 0 or new_pos[1] >= self.reward.shape[1]:
            reward = self.cliff
            observation = agent.set_pos(agent.pos)
            done = True
        else:
            observation = agent.set_pos(new_pos)
            reward = self.reward[observation[0],observation[1]]

        return observation, reward, done



def finding_optimal_value_function(env, agent, v_table):
    k = 1
    discount_factor = 0.9
    while(True):
        delta=0
        temp_v = copy.deepcopy(v_table)


        for i in range(env.reward.shape[0]):
            for j in range(env.reward.shape[1]):
                temp = -1e+10
                for action in range(len(agent.action)):
                    agent.set_pos([i,j])
                    observation, reward, done = env.move(agent, action)
                    if temp < reward + discount_factor*v_table[observation[0],observation[1]]:
                        temp = reward + discount_factor*v_table[observation[0],observation[1]]
                v_table[i,j] = temp


        delta = np.max([delta, np.max(np.abs(temp_v-v_table))])

        if delta < 0.0000001:
            break

        print("V{0}(S) : k = {1:3d}    delta = {2:0.6f}".format(k,k, delta))
        show_v_table(np.round(v_table,2),env)
        k +=1

    return v_table

def policy_extraction(env, agent, v_table, optimal_policy):

    discount_factor = 0.9

    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            temp =  -1e+10
            for action in range(len(agent.action)):
                agent.set_pos([i,j])
                observation, reward, done = env.move(agent,action)
                if temp < reward + discount_factor * v_table[observation[0],observation[1]]:
                    optimal_policy[i,j] = action
                    temp = reward + discount_factor * v_table[observation[0],observation[1]]

    return optimal_policy


np.random.seed(0)
env = Environment()
agent = Agent()

v_table =  np.random.rand(env.reward.shape[0], env.reward.shape[1])

print("Initial random V0(S)")
show_v_table(np.round(v_table,2),env)
print()

optimal_policy = np.zeros((env.reward.shape[0], env.reward.shape[1]))

print("start Value iteration")
print()

start_time = time.time()
v_table = finding_optimal_value_function(env, agent, v_table)
optimal_policy = policy_extraction(env, agent, v_table, optimal_policy)


print("total_time = {}".format(np.round(time.time()-start_time),2))
print()
print("Optimal policy")
show_policy(optimal_policy, env)
