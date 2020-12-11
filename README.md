# tum-adlr-ws20-08

Advanced Deep Learning for Robotics project: Evaluation of End-To-End as well as Hybrid Control approaches for Fixed Wing UAVs

# Setup

Clone this repo with the option --recurse-submodules
Create a new environment with the provided env.yml or use the requirements.txt to install all needed packages
Please use Python version 3.7

1. With env.yml
   conda env create --file envname.yml
2. Using requirements.txt
   Use or activate desired environment
   pip install -r requirements.txt

# Installation

First install all related code which is under the "papers" folder

1. fixed-wing-gym
   cd papers/fixed-wing-gym/
   pip install -e .

2. pyfly
   cd papers/pyfly/
   pip install -e .

3. Oyster
   cd papers/oyster/rand_param-envs
   pip install -e .

To install locally, you will need to first install [MuJoCo](https://www.roboti.us/index.html)
For the task distributions in which the reward function varies (Cheetah, Ant, Humanoid), install MuJoCo200.
Set `LD_LIBRARY_PATH` to point to both the MuJoCo binaries (`/$HOME/.mujoco/mujoco200/bin`) as well as the gpu drivers (something like `/usr/lib/nvidia-390`, you can find your version by running `nvidia-smi`).

For the task distributions where different tasks correspond to different model parameters (Walker and Hopper), MuJoCo131 is required.
Simply install it the same way as MuJoCo200.
These environments make use of the module `rand_param_envs` which is submoduled in this repository.
Add the module to your python path, `export PYTHONPATH=./rand_param_envs:$PYTHONPATH`
(Check out [direnv](https://direnv.net/) for handy directory-dependent path managenement.)
