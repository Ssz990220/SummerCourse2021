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
import numpy as np
import random
import torch
import cv2
import time
import argparse
import pygame

env = make('snakes_1v1', conf=None)
## Game Init ##
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

def get_state(all_observation, id):
    return all_observation[id]

def get_keyboard_action(available_actions=[0,1,2,3]):
    # 0 = up 1=down 2=left 3=right
    while(1):
        k = cv2.waitKeyEx(20)
        # print(k)
        if k==2490368: # UP
            action = 0
        elif k==2621440: # DOWN
            action = 1
        elif k==2555904: # RIGHT
            action = 3
        elif k==2424832: # LEFT
            action = 2
        else:
            continue
        if action in available_actions:break
    return action
      
                 
def main():
    ## Play mode ##
    parser = argparse.ArgumentParser()

    parser.add_argument('--play_mode', default='EvP')

    mode = parser.parse_args()
    
    

    agent_trained_index = 2  # in env
    agent_copied_index = 3
    episode = 0
    ctrl_agent_index = [0]  # in code
    ctrl_agent_num = len(ctrl_agent_index)
    model = DQN_COPY(obs_dim, action_dim, args.hidden_size)
    run_dir = 'H:\\Project\\SummerCourse2021\\course_competition\\rl_trainer\\models\\snake1v1\\run12'
    model2 = DQN_COPY(obs_dim, action_dim, args.hidden_size)
    model.load(run_dir, 100000)
    model2.load(run_dir, 150000)
    while episode < args.max_episodes:
        state = env.reset()
        img = env.render_board()
        board_render = cv2.imshow('board', img)
        cv2.waitKey(10)
        time.sleep(0.5)
        state_rl_agent_controlled = get_state(state, 0)
        state_rl_agent_copy = get_state(state, 1)
        obs = get_observations(state_rl_agent_controlled, agent_trained_index, obs_dim)
        obs2 = get_observations(state_rl_agent_copy, agent_copied_index, obs_dim)
        
        episode += 1
        step = 0
        episode_reward = np.zeros(2)
        
        while True:
            
            ## Choose Action ##
            if mode.play_mode == 'EvE':
                action = model.choose_action(obs)
                action2 = model2.choose_action(obs2)
            elif mode.play_mode == 'EvR':
                action = model.choose_action(obs)
                action2 = np.random.randint(action_dim)
            elif mode.play_mode == 'EvG':
                action = model.choose_action(obs)
                action2 = get_greedy(state[1])[0]
            elif mode.play_mode == 'EvP':
                action = model.choose_action(obs)
                action2 = get_keyboard_action()
            elif mode.play_mode == "PvR":
                action2 = get_keyboard_action()
                action = np.random.randint(action_dim)
            elif mode.play_mode == "PvG":
                action2 = get_keyboard_action()
                action = get_greedy(state[1])[0]
            else:
                raise ValueError("No such play mode! \n")
            
            actions = np.array([action, action2])

            next_state, reward, done, _, _ = env.step(env.encode(actions))
            img = env.render_board()
            board_render = cv2.imshow('board', img)
            cv2.waitKey(10)
            # time.sleep(0.5)
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
                next_obs = get_observations(next_state_rl_agent, agent_trained_index, obs_dim)
                next_obs2 = get_observations(next_state_rl_agent2, agent_copied_index, obs_dim)
            done = np.array([done] * ctrl_agent_num)
            obs = next_obs
            obs2 = next_obs2
            state = next_state
            step += 1
            if args.episode_length <= step or (True in done):
                print(f'[Episode {episode:05d}] score: {episode_reward[0]} reward: {step_reward[0]:.2f}')
                env.reset()
                break
            
if __name__=='__main__':
    main()