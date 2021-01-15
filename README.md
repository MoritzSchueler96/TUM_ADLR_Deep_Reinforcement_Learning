# tum-adlr-ws20-08

Advanced Deep Learning for Robotics project: Evaluation of End-To-End as well as Hybrid Control approaches for Fixed Wing UAVs

# Setup

Clone this repo
Create a new environment with the provided env.yml _magpie-env.yml_ or use the requirements.txt _requirements-magpie.txt_ to install all needed packages
Please use Python version 3.8.3

1. With env.yml
   conda env create --file magpie-env.yml
2. Using requirements.txt
   Use or activate desired environment
   pip install -r requirements-magpie.txt

# Installation

First install all related code which is under the "magpie/libs" folder

1.  fixed-wing-gym
    cd magpie/libs/fixed-wing-gym/
    pip install -e .

2.  pyfly
    cd magpie/libs/pyfly/
    pip install -e .

3.  pyfly fixed wing visualizer
    cd magpie/libs/pyfly-fixed-wing-visualizer
    pip install -e .

4.  Pearlite
    a.
    cd magpie/libs/pearlite/
    cp -ar . _PATH/TO/YOUR/CONDA/ENV_/lib/python3.8/site-packages/stable_baselines3

        for example:
        cp -ar . /home/user/miniconda3/envs/magpie/lib/python3.8/site-packages/stable_baselines3

    b.
    cd magpie/libs/stable-baselines3/
    pip install -e .
