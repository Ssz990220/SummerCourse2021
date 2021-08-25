import torch.nn as nn

# ====================================== helper functions ======================================
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from common import make_grid_map, get_surrounding, get_observations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# ====================================== define algo ===========================================
# todo
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = state_dim
        self.output_size = action_dim
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# todo
class DQN(object):
    def __init__(self):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_dim
        self.critic_net = Critic()

    def choose_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
        action = torch.argmax(self.critic_net(observation)).item()
        return action
    def load(self, file):
        self.critic_net.load_state_dict(torch.load(file))


def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        joint_action.append(one_hot_action)
    return joint_action


# ===================================== define agent =============================================
#todo
state_dim = 18
action_dim = 4
hidden_dim = 256
agent = DQN()
critic_net_file = os.path.dirname(os.path.abspath(__file__)) + '/critic.pth'
agent.load(critic_net_file)


# ================================================================================================
"""
input:
    observation: dict
    {
        1: 豆子，
        2: 第一条蛇的位置，
        3：第二条蛇的位置，
        "board_width": 地图的宽，
        "board_height"：地图的高，
        "last_direction"：上一步各个蛇的方向，
        "controlled_snake_index"：当前你控制的蛇的序号（2或3）
        }
return: 
    action: eg. [[[0,0,0,1]]]
"""
# todo
def get_state(all_observation):
    return all_observation[0] # todo

def my_controller(observation, action_space_list, is_act_continuous=False):
    agent_trained_index = observation['controlled_snake_index']
    obs = get_observations(observation, agent_trained_index, 18)
    action = agent.choose_action(obs)
    return to_joint_action(action, 2)
