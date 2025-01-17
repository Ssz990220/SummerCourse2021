from common import *
from arguments import get_args
from log_path import make_logpath
from collections import namedtuple
from dqn import DQN
from dqn_copy import DQN_COPY
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from env.chooseenv import make
from tensorboardX import SummaryWriter

import numpy as np
import random
import torch
import cv2
import time

env = make('snakes_1v1', conf=None)

args = get_args()

game_name = args.game_name
print(f'game name: {args.game_name}')

width = env.board_width
print(f'Game board width: {width}')
height = env.board_height
print(f'Game board height: {height}')
action_dim = env.get_action_dim()
print(f'action dimension: {action_dim}')
obs_dim = 18
print(f'observation dimension: {obs_dim}')


def get_players_and_action_space_list():
    n_agent_num = list(env.agent_nums)
    for i in range(1, len(n_agent_num)):
        n_agent_num[i] += n_agent_num[i - 1]

    players_id = []
    actions_space = []
    for policy_i in range(len(env.obs_type)):
        if policy_i == 0:
            players_id_list = range(n_agent_num[policy_i])
        else:
            players_id_list = range(n_agent_num[policy_i - 1], n_agent_num[policy_i])
        players_id.append(players_id_list)

        action_space_list = [env.get_single_action_space(player_id) for player_id in players_id_list]
        actions_space.append(action_space_list)

    return players_id, actions_space


def get_state(all_observation, id):
    return all_observation[id] # todo


def main(args):
    # set seed
    torch.manual_seed(args.seed_nn)
    np.random.seed(args.seed_np)
    random.seed(args.seed_random)

    # 定义保存路径
    run_dir, log_dir = make_logpath(game_name, args.algo)
    writer = SummaryWriter(str(log_dir))

    # 保存训练参数 以便复现
    if args.train_redo:
        config_dir = os.path.join(os.path.dirname(log_dir), 'run%i' % (args.run_redo))
        load_config(args, config_dir)
        save_config(args, log_dir)
    else:
        save_config(args, log_dir)

    ctrl_agent_index = [0]  # in code
    ctrl_agent_num = len(ctrl_agent_index)
    agent_trained_index = 2  # in env
    agent_copied_index = 3

    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
    model = DQN(obs_dim, action_dim, ctrl_agent_num, args)
    model2 = DQN_COPY(obs_dim, action_dim, args.hidden_size)

    episode = 0
    is_init = True
    
    while episode < args.max_episodes:
        state = env.reset()

        state_rl_agent_controlled = get_state(state, 0)
        state_rl_agent_copy = get_state(state, 1)

        obs = get_observations(state_rl_agent_controlled, agent_trained_index, obs_dim)
        obs2 = get_observations(state_rl_agent_copy, agent_copied_index, obs_dim)

        episode += 1
        step = 0
        episode_reward = np.zeros(2)

        while True:
            action = model.choose_action(obs)
            if is_init:
                action2 = get_greedy(state[1])[0]
                actions = np.array([action, action2])
            else:
                action2 = model2.choose_action(obs2)
                actions = np.array([action, action2])

            next_state, reward, done, _, _ = env.step(env.encode(actions))
            if args.render:
                img = env.render_board()
                board_render = cv2.imshow('board', img)
                cv2.waitKey(10)
                time.sleep(0.5)

            next_state_rl_agent = get_state(next_state, 0)
            next_state_rl_agent2 = get_state(next_state, 1)

            reward = np.array(reward)
            episode_reward += reward

            if done:
                if episode_reward[0] > episode_reward[1]:
                    step_reward = get_reward(next_state_rl_agent, ctrl_agent_index, reward, final_result=1)
                elif episode_reward[0] < episode_reward[1]:
                    step_reward = get_reward(next_state_rl_agent, ctrl_agent_index, reward, final_result=2)
                else:
                    step_reward = get_reward(next_state_rl_agent, ctrl_agent_index, reward, final_result=3)
                next_obs = np.zeros((ctrl_agent_num, obs_dim))
                next_obs2 = np.zeros((ctrl_agent_num, obs_dim))
            else:
                step_reward = get_reward(next_state_rl_agent, ctrl_agent_index, reward, final_result=0)
                next_obs = get_observations(next_state_rl_agent, agent_trained_index, obs_dim)
                next_obs2 = get_observations(next_state_rl_agent2, agent_copied_index, obs_dim)

            done = np.array([done] * ctrl_agent_num)

            # store transitions
            trans = Transition(obs, action, step_reward, np.array(next_obs), done)
            model.store_transition(trans)
            model.learn()
            obs = next_obs
            obs2 = next_obs2
            state = next_state
            step += 1

            if args.episode_length <= step or (True in done):
                print(f'[Episode {episode:05d}] score: {episode_reward[0]} reward: {step_reward[0]:.2f}')

                reward_tag = 'reward'
                loss_tag = 'loss'
                writer.add_scalars(reward_tag, global_step=episode,
                                   tag_scalar_dict={'score': episode_reward[0], 'reward': step_reward[0]})
                if model.loss:
                    writer.add_scalars(loss_tag, global_step=episode,
                                       tag_scalar_dict={'loss': model.loss})
                    print(f'\t\t\t\tloss {model.loss:.3f}')
                if episode % args.view_interval == 1:
                    args.render = False
                    cv2.destroyAllWindows()
                if episode % args.view_interval == 0:
                    # args.render = True
                    pass
                if episode % args.save_interval == 0:
                    model.save(run_dir, episode)
                    # args.render = True
                    if episode % args.update_policy_freq == 0:
                        model2.load(run_dir, episode)
                        is_init = False
                    

                env.reset()
                break


if __name__ == '__main__':
    main(args)
