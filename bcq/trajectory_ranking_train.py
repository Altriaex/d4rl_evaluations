import argparse
from email.policy import default
import gym
from matplotlib.pyplot import flag
import numpy as np
import os
import torch
import shutil
import time
import d4rl
import json

import continuous_bcq.BCQ
import continuous_bcq.utils as utils
import os.path as osp
from absl import logging, app, flags

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
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        replay_buffer.add(obs, action, new_obs, reward, done_bool)
    logging.info(f'Loaded buffer from {h5path}')
    
    evaluations = []
    training_iters = 0
    while training_iters < args.max_timesteps: 
        logging.info(f"Training step: {training_iters}")
        tstart = time.time()
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        tend = time.time()
        evaluations.append(eval_policy(policy, game, args.seed))
        np.save(os.path.join(output_dir, f"eval_returns.npy"), evaluations)
        training_iters += args.eval_freq
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
    logging.info(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward
flags.DEFINE_integer('seed', 0, "Sets Gym, PyTorch and Numpy seeds")
flags.DEFINE_string('buffer_name', "Robust", "Prepends name to filename")
flags.DEFINE_integer('eval_freq', 5000, "How often (time steps) we evaluate")
flags.DEFINE_integer('max_timesteps', 1000000, "Max time steps to run environment or train for (this defines buffer size)")

flags.DEFINE_integer("start_timesteps", 25000, "Time steps initial random policy is used before training behavioral")
flags.DEFINE_float("rand_action_p", 0.3, "Probability of selecting random action during batch generation")
flags.DEFINE_float("gaussian_std", 0.3, "Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)")
flags.DEFINE_integer("batch_size", 100, "Mini batch size") 
flags.DEFINE_float("discount", 0.99, "Discount factor")     
flags.DEFINE_float("tau", 0.005, "Target network update rate")
flags.DEFINE_float("lmbda", 0.75, "Weighting for clipped double Q-learning in BCQ")
flags.DEFINE_float("phi", 0.05, "Max perturbation hyper-parameter for BCQ")
flags.DEFINE_string("exp_dir", None, "path to experiments")
flags.DEFINE_string("method", None, "the reward learning method")

args = flags.FLAGS

def main(argv):
    logging.set_verbosity(logging.INFO)
    path, split = osp.split(args.exp_dir)
    path, game = osp.split(path)
    exp_base, exp_id = osp.split(path)
    data_folder_name = "_".join(["training_logs", "BCQ", args.method])
    data_path = osp.join(args.exp_dir, data_folder_name)
    if osp.exists(data_path):
        shutil.rmtree(data_path)    
    os.makedirs(data_path, exist_ok=True)
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

if __name__ == '__main__':
    flags.mark_flag_as_required('method')
    flags.mark_flag_as_required('exp_path')
    app.run(main)