# Understanding and Adopting Rational Behavior by Bellman Score Estimation

Author: Kuno Kim 

This repo contains the official implementation for the [ICLR submission](https://tinyurl.com/y7byr8rx)

**Note**: ** This repo is being actively updated. **

-------------------------------------------------------------------------------------

## Main Dependencies

* PyTorch

* cudatoolkit

* PyYAML

* hydra

* dm_control

## Requirements
We assume you have access to a gpu that can run CUDA 9.2 or above. We used `pytorch==1.10.1` with `cudatoolkit==11.3.1` in our experiments. The simplest way to install all required dependencies is to create an anaconda environment and activate it:
```
conda env create -f conda_env.yaml
source activate gac
```
Next, install the `ReparamModule` package following the instructions from [here](https://github.com/SsnL/PyTorch-Reparam-Module).


## Running Experiments

### Project Structure

`train_gac.py` is the common gateway to all experiments.

```bash
usage: train_gac.py env=ENV_NAME
                    experiment=EXP_NAME
                    seed=SEED
                    load_demo_path=PATH_TO_DEMO
                    load_expert_path=PATH_TO_EXPERT
                    num_transitions=NUM_DEMO

optional arguments:
  experiment          Name of experiment for logging purposes
  seed                Random seed
  load_demo_path      Global path to the saved expert demonstrations
  load_expert_path    Global path to the saved expert (only for evaluation purposes)
  num_transitions     Number of demonstrations to use
```

Configuration files are stored in  `config/`. For example, the configuration file of `GAC` is `config/imitate.yaml` and `config/agent/gac.yaml`. Log files are commonly stored in `exp/` including the tensorboard files.

### Training

Download the [expert demonstations](https://tinyurl.com/5acd9kz7) and place them in `gac/saved_demo`. Each pickle file contains 1000 demonstration trajectories for a different environment. The environment names match the file names. The usage of `train_gac.py` is quite self-evident. For example, we can train GAC for the `walker_walk` task with one demonstration by running

```bash
python train_gac.py env='walker_walk' experiment='walker_walk' seed=0 load_demo_path=/user/gac/saved_demo/walker_walk.pickle load_expert_path=/user/gac/saved_experts/walker_walk.pt num_transitions=1
```

Choose from a variety of environments `walker_stand, walker_walk, hopper_stand, cheetah_run, quadruped_run`.


### Evaluation
Running `train_gac.py` outputs evaluation metrics to the console. The long names for the shorthand acroynms can be found in `logger.py`. For the evaluation step outputs, `L_R` shows the average learner episode reward which quantifies control performance of the learner. Another convenient way to monitor training progress is to use tensorboard. For example, to visualize the runs started on 2022.10.01, one may run

```bash
tensorboard --logdir exp/2022.10.01 --port 8008
```

The evaluation metrics are then found at `http://localhost:8008`. The "learner_episode_reward" graph shows the average episode reward obtained during the evaluation step. A sample learning curve for the `walker_walk` task should look like so.

<img src="/figures/learning_curve.png" width="400" />




