# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
from absl import app
from absl import flags
from absl import logging
import shutil

import gin
import tensorflow as tf0
import tensorflow.compat.v1 as tf

from behavior_regularized_offline_rl.brac import agents
from behavior_regularized_offline_rl.brac import train_eval_offline
from behavior_regularized_offline_rl.brac import utils

import d4rl

tf0.compat.v1.enable_v2_behavior()


# Flags for offline training.
flags.DEFINE_string('exp_path',
                    os.path.join(os.getenv('HOME', '/'), 'tmp/offlinerl/learn'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('agent_name', 'brac_primal', 'agent name.')
flags.DEFINE_integer('total_train_steps', int(1e6), '')
flags.DEFINE_integer('bc_train_steps', int(1e6), '')
flags.DEFINE_integer('n_train', int(10e6), '')
flags.DEFINE_integer('n_eval_episodes', 10, '')
flags.DEFINE_integer('value_penalty', 1, '')
flags.DEFINE_integer('save_freq', 5000, '')
flags.DEFINE_float('alpha', 1.0, '')
flags.DEFINE_float('discount', 0.99, '')
flags.DEFINE_string('method', 'bt', '')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')
flags.DEFINE_bool('train_alpha', False, '')

FLAGS = flags.FLAGS

def train_bc():
    path, split = osp.split(FLAGS.exp_path)
    path, game = osp.split(path)
    exp_base, exp_id = osp.split(path)
    data_folder_name = "_".join(["training_logs", "BRAC", FLAGS.method])
    data_path = osp.join(FLAGS.exp_path, data_folder_name)
    datafile = osp.join(data_path, game + ".hdf5")
    log_dir = os.path.join(
          FLAGS.exp_path,
          "BC")
    model_arch = ((200,200),)
    opt_params = (('adam', 5e-4),)
    utils.maybe_makedirs(log_dir)
    train_eval_offline.train_eval_offline(
      log_dir=log_dir,
      data_file=datafile,
      agent_module=agents.AGENT_MODULES_DICT['bc'],
      env_name=game,
      n_train=FLAGS.n_train,
      total_train_steps=FLAGS.bc_train_steps,
      n_eval_episodes=1,
      model_params=model_arch,
      optimizers=opt_params,
      seed=int(split),
      use_seed_for_data=True)
    return log_dir

def main(_):
    path, split = osp.split(FLAGS.exp_path)
    path, game = osp.split(path)
    exp_base, exp_id = osp.split(path)
    data_folder_name = "_".join(["training_logs", "BRAC", FLAGS.method])
    data_path = osp.join(FLAGS.exp_path, data_folder_name)
    if osp.exists(data_path):
        shutil.rmtree(data_path)    
    os.makedirs(data_path, exist_ok=True)
    data_zipfile = "_".join([exp_id, game, split, FLAGS.method]) + ".zip"
    shutil.copy2(osp.join(exp_base, "preference_rewards", data_zipfile), data_path)
    shutil.unpack_archive(osp.join(data_path, data_zipfile), extract_dir=data_path, format="zip")
    os.remove(osp.join(data_path, data_zipfile))
    datafile = osp.join(data_path, game + ".hdf5")
    logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

    bc_log_dir = train_bc()
    behavior_ckpt_file = os.path.join(bc_log_dir, 'agent_behavior')

    log_dir = os.path.join(FLAGS.exp_path, "BRAC_" + FLAGS.method)

    model_arch = (((300, 300), (200, 200),), 5)
    opt_params = (('adam', 1e-3), ('adam', 3e-4), ('adam', 1e-5))

    utils.maybe_makedirs(log_dir)
    train_eval_offline.train_eval_offline(
      log_dir=log_dir,
      data_file=datafile,
      agent_module=agents.AGENT_MODULES_DICT[FLAGS.agent_name],
      env_name=game,
      n_train=FLAGS.n_train,
      total_train_steps=FLAGS.total_train_steps,
      n_eval_episodes=FLAGS.n_eval_episodes,
      model_params=model_arch,
      optimizers=opt_params,
      behavior_ckpt_file=behavior_ckpt_file,
      value_penalty=bool(FLAGS.value_penalty),
      save_freq=FLAGS.save_freq,
      alpha=FLAGS.alpha,
      seed=int(split),
      use_seed_for_data=True,
      discount=FLAGS.discount,
      train_alpha=FLAGS.train_alpha,
      summary_freq=1000)
        
    archive_name = osp.join(
        exp_base, "agents", "_".join([exp_id, game, split, "BRAC", FLAGS.method]))
    shutil.make_archive(
        base_name=archive_name,
        root_dir=log_dir,
        base_dir=None, format="zip")
    archive_name = osp.join(
        exp_base, "agents", "_".join([exp_id, game, split, "BC"]))
    shutil.make_archive(
        base_name=archive_name,
        root_dir=bc_log_dir,
        base_dir=None, format="zip")
    shutil.rmtree(data_path)


if __name__ == '__main__':
    app.run(main)
