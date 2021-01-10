# Setup

Clone this repo
Create a new environment with the provided env.yml _pearlite-env.yml_ or use the requirements.txt _requirements-pearlite.txt_ to install all needed packages
Please use Python version 3.8.3

1. With env.yml
   conda env create --file pearlite-env.yml
2. Using requirements.txt
   Use or activate desired environment
   pip install -r requirements-pearlite.txt

# Installation

Install pearlite code which is under the "magpie/libs" folder

1. Pearlite
   cd magpie/libs/pearlite/
   cp -ar . _PATH/TO/YOUR/CONDA/ENV_/lib/python3.8/site-packages/stable_baselines3

    for example:
    cp -ar . /home/user/miniconda3/envs/magpie/lib/python3.8/site-packages/stable_baselines3
