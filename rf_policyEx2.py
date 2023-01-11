import numpy as np
import time

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



# 정책 policy 화살표로 그리기
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
    # 행동에 따른 에이전트의 좌표 이동(위, 오른쪽, 아래, 왼쪽)
    action = np.array([[-1,0],[0,1],[1,0],[0,-1]])

    # 각 행동별 선택확률
    select_action_pr = np.array([0.25,0.25,0.25,0.25])

    # 에이전트의 초기 위치 저장
    def __init__(self):
        self.pos = (0,0)

    # 에이전트의 위치 저장
    def set_pos(self, position):
        self.pos = position
        return self.pos

    # 에이전트의 위치 불러오기
    def get_pos(self):
        return self.pos


class Environment():
    # 미로밖(절벽), 길, 목적지와 보상 설정
    cliff = -3
    road = -1
    goal = 1

    # 목적지 좌표 설정
    goal_position = [2,2]

    # 보상 리스트 숫자
    reward_list = [[road,road,road],
                   [road,road,road],
                   [road,road,goal]]

    # 보상 리스트 문자
    reward_list1 = [["road","road","road"],
                    ["road","road","road"],
                    ["road","road","goal"]]

    # 보상 리스트를 array로 설정
    def __init__(self):
        self.reward = np.asarray(self.reward_list)

    # 선택된 에이전트의 행동 결과 반환 (미로밖일 경우 이전 좌표로 다시 복귀)
    def move(self, agent, action):

        done = False

        # 행동에 따른 좌표 구하기
        new_pos = agent.pos + agent.action[action]

        # 현재좌표가 목적지 인지확인
        if self.reward_list1[agent.pos[0]][agent.pos[1]] == "goal":
            reward = self.goal
            observation = agent.set_pos(agent.pos)
            done = True
        # 이동 후 좌표가 미로 밖인 확인
        elif new_pos[0] < 0 or new_pos[0] >= self.reward.shape[0] or new_pos[1] < 0 or new_pos[1] >= self.reward.shape[1]:
            reward = self.cliff
            observation = agent.set_pos(agent.pos)
            done = True
        # 이동 후 좌표가 길이라면
        else:
            observation = agent.set_pos(new_pos)
            reward = self.reward[observation[0],observation[1]]

        return observation, reward, done #next state, done= arrive or cliff



# 반복 정책 평가
np.random.seed(0)
env = Environment()
agent = Agent()
gamma = 0.9

# 𝑉(𝑠)=0으로 초기화
v_table = np.zeros((env.reward.shape[0],env.reward.shape[1]))

print("start Iterative Value Evaluation")

k = 1
print()
print("V0(S)   k = 0")

# 초기화된 V 테이블 출력
show_v_table(np.round(v_table,2),env)

# 시작 시간 변수에 저장
start_time = time.time()


def policy_evalution(env, agent, v_table, policy):
    discount_factor = 0.9

    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            # 에이전트를 지정된 좌표에 위치시킨후 가치함수를 계산
            agent.set_pos([i,j])
            # 현재 정책의 행동을 선택
            action = policy[i,j]
            observation, reward, done = env.move(agent, action)
            v_table[i,j] = reward + discount_factor * v_table[observation[0],observation[1]]

    return v_table

def policy_improvement(env, agent, v_table, policy):

    discount_factor = 0.9

    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            # 가능한 행동중 최댓값을 가지는 행동을 선택
            temp_action = 0
            temp_value =  -1e+10
            for action in range(len(agent.action)):
                agent.set_pos([i,j])
                observation, reward, done = env.move(agent,action)
                if temp_value < reward + discount_factor * v_table[observation[0],observation[1]]:
                    temp_action = action
                    temp_value = reward + discount_factor * v_table[observation[0],observation[1]]

            policy[i,j] = temp_action
    return policy


def value_evaluation(env, agent, v_table, a_table):
    discount_factor = 0.9

    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            temp_action = 0
            temp_value = -1e+10

            for action in range(len(agent.action)):
                agent.set_pos([i, j])
                observation, reward, done = env.move(agent, action)
                if temp_value < reward + discount_factor * v_table[observation[0], observation[1]]:
                    temp_action = action
                    temp_value = reward + discount_factor * v_table[observation[0], observation[1]]

            a_table[i, j] = temp_action
            v_table[i, j] = temp_value

    return v_table, a_table

# def get_action(env, agent, v_table):


np.random.seed(0)
env = Environment()
agent = Agent()


v_table =  np.random.rand(env.reward.shape[0], env.reward.shape[1])
a_table =  np.random.rand(env.reward.shape[0], env.reward.shape[1])

policy = np.random.randint(0, 4,(env.reward.shape[0], env.reward.shape[1]))

print("Initial random V(S)")
show_v_table(np.round(v_table,2),env)
print()
print("Initial random Policy π0(S)")
show_policy(policy,env)
print("start policy iteration")

# 시작 시간을 변수에 저장
start_time = time.time()

max_iter_number = 50
for iter_number in range(max_iter_number):

    #정책평가
    # v_table= policy_evalution(env, agent, v_table, policy)
    v_table, a_table = value_evaluation(env, agent, v_table, a_table)
    # 정책 평가 후 결과 표시
    print("")
    print("V:{0:}".format(iter_number))
    show_v_table(np.round(v_table,2),env)
    print()

    #정책개선
    # policy = policy_improvement(env, agent, v_table, policy)

    # policy 변화 저장
    print("policy:{}".format(iter_number+1))
    show_policy(policy,env)


print("total_time = {}".format(time.time()-start_time))
