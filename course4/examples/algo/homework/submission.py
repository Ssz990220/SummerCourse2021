# # This is homework.
# # Load your model and submit this to Jidi

import torch
import os
import numpy as np

# load critic
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from critic import Critic


# TODO
class IQL:
    def __init__(self):
        self.critic_net = Critic(18,4,64)
    
    def choose_action(self, obs):
        # actions = []
        # for i in range(len(obs)):
        #     observation = torch.tensor(obs[i], dtype=torch.float).view(1, -1)
        #     action = torch.argmax(self.critic_net(observation)).item()
        #     actions.append(action)
        # return actions
        observation = torch.tensor(obs, dtype=torch.float).view(1, -1)
        action = torch.argmax(self.critic_net(observation)).item()
        return action

    
    def load(self, file):
        self.critic_net.load_state_dict(torch.load(file))


#TODO
def action_from_algo_to_env(joint_action):
    # joint_action_ = []
    # for i in range(2):
    #     action_a = joint_action[i]
    #     each = [0] * 4
    #     each[action_a] = 1
    #     joint_action_.append([each])
    # return joint_action_
    action_a = joint_action
    each = [0] * 4
    each[action_a] = 1
    return each


# todo
# Once start to train, u can get saved model. Here we just say it is critic.pth.
critic_net = os.path.dirname(os.path.abspath(__file__)) + '/critic_10000.pth'
agent = IQL()
agent.load(critic_net)
# demo_state_1 = list(np.random.rand(1,18))
# demo_state_2 = list(np.random.rand(1,18))
# demo_state = [demo_state_1, demo_state_2]
# actions = agent.choose_action(demo_state_1)
# print(action_from_algo_to_env(actions))
# todo

def my_controller(observation, action_space, is_act_continuous=False):
    obs = observation['obs']
    action = agent.choose_action(obs)
    return action_from_algo_to_env(action)