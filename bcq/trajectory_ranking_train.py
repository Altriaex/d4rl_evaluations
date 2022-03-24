import argparse
import gym
import numpy as np
import os
import torch
import shutil

import d4rl
import json

import continuous_bcq.BCQ
import continuous_bcq.utils as utils
import os.path as osp

# Trains BCQ offline
def train_BCQ(env, state_dim, action_dim, max_action, device, output_dir, args):
    path, split = osp.split(args.exp_dir)
    path, game = osp.split(path)
    # For saving files

    # Initialize policy
    policy = continuous_bcq.BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    data_folder_name = "_".join(["training_logs", "BCQ", args.method])
    data_path = osp.join(args.exp_dir, data_folder_name)
    h5path = osp.join(data_path, game + ".hdf5")
    dataset = env.get_dataset(h5path=h5path)
    N = dataset['rewards'].shape[0]
    print('Loading buffer!')
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        replay_buffer.add(obs, action, new_obs, reward, done_bool)
    print('Loaded buffer')
    
    evaluations = []
    training_iters = 0
    while training_iters < args.max_timesteps: 
        print('Train step:', training_iters)
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        evaluations.append(eval_policy(policy, game, args.seed))
        np.save(os.path.join(output_dir, f"eval_returns.npy"), evaluations)
        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")
        policy.save(output_dir)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
parser.add_argument("--eval_freq", default=5e3, type=float)     # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)
parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used before training behavioral
parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
parser.add_argument("--discount", default=0.99)                 # Discount factor
parser.add_argument("--tau", default=0.005)                     # Target network update rate
parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ
parser.add_argument("--exp_dir", default="", type=str)
parser.add_argument("--method", default="", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    path, split = osp.split(args.exp_dir)
    path, game = osp.split(path)
    exp_base, exp_id = osp.split(path)
    data_folder_name = "_".join(["training_logs", "BCQ", args.method])
    data_path = osp.join(args.exp_dir, data_folder_name)
    if osp.exists(data_path):
        shutil.rmtree(data_path)    
    data_file = "_".join([exp_id, game, split, args.method]) + ".zip"
    shutil.copy2(osp.join(exp_base, "preference_rewards", data_file), data_path)
    shutil.unpack_archive(osp.join(data_path, data_file), extract_dir=data_path, format="zip")
    os.remove(osp.join(data_path, data_file))
    results_dir = os.path.join(args.exp_dir, f'BCQ_{args.method}')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'params.json'), 'w') as params_file:
        json.dump({'game': game, 'seed': args.seed}, params_file)
    env = gym.make(game)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_BCQ(env, state_dim, action_dim, max_action, device, results_dir, args)
    shutil.rmtree(data_path)

    archive_name = osp.join(exp_base, "agents", "_".join([exp_id, game, split, "BCQ", args.method]))
    shutil.make_archive(base_name=archive_name,
        root_dir=results_dir, base_dir=None, format="zip")