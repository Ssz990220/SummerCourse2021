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
    return [each]


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
    obs = get_observations(observation, observation['controlled_snake_index'],18)
    action = agent.choose_action(obs)
    return action_from_algo_to_env(action)

def get_observations(state, agent_trained_index, obs_dim):
    state_copy = state.copy()

    agents_index = state_copy["controlled_snake_index"]

    if agents_index != agent_trained_index:
        error = "训练的智能体：{name}, 观测的智能体：{url}".format(name=agents_index, url=agent_trained_index)
        raise Exception(error)

    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3}}
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state = np.array(snake_map)
    state = np.squeeze(snake_map, axis=2)

    snakes_position = np.array(snakes_positions[agents_index], dtype=object)

    beans_position = np.array(beans_positions).flatten()

    observations = np.zeros((1, obs_dim)) # todo

    # self head position
    observations[0][:2] = snakes_position[0][:]

    # head surroundings
    head_x = snakes_position[0][1]
    head_y = snakes_position[0][0]
    head_surrounding = get_surrounding(state, board_width, board_height, head_x, head_y)
    observations[0][2:6] = head_surrounding[:]

    # beans positions
    observations[0][6:16] = beans_position[:]

    # other snake head positions
    snakes_other_position = np.array(snakes_positions[3], dtype=object) # todo
    observations[0][16:] = snakes_other_position[0][:]

    return observations

def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map

def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding
