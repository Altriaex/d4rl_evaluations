# coding=utf-8

"""Offline training binary."""
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
flags.DEFINE_integer('total_train_steps', int(1e6), '')
flags.DEFINE_integer('n_train', int(10e6), '')
flags.DEFINE_integer('value_penalty', 0, '')
flags.DEFINE_integer('save_freq', 1000, '')
flags.DEFINE_float('alpha', 1.0, '')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS

def main(_):
    # Setup log dir.
    path, split = osp.split(FLAGS.exp_path)
    path, game = osp.split(path)
    exp_base, exp_id = osp.split(path)
    log_dir = os.path.join(
          FLAGS.exp_path,
          "BC")
    model_arch = ((200,200),)
    opt_params = (('adam', 5e-4),)
    utils.maybe_makedirs(log_dir)
    train_eval_offline.train_eval_offline(
      log_dir=log_dir,
      data_file=None,
      agent_module=agents.AGENT_MODULES_DICT['bc'],
      env_name=game,
      n_train=FLAGS.n_train,
      total_train_steps=FLAGS.total_train_steps,
      n_eval_episodes=10,
      model_params=model_arch,
      optimizers=opt_params,
      seed=int(split),
      use_seed_for_data=True)
    archive_name = osp.join(
        exp_base, "agents", "_".join([exp_id, game, split, "BC"]))
    shutil.make_archive(
        base_name=archive_name,
        root_dir=log_dir,
        base_dir=None, format="zip") 

if __name__ == '__main__':
    app.run(main)
