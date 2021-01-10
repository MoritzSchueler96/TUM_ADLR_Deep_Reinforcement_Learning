# Setup

Clone this repo
Create a new environment with the provided env.yml _fixed-wing-env.yml_ or use the requirements.txt _requirements-fixed-wing.txt_ to install all needed packages
Please use Python version 3.5.6

1. With env.yml
   conda env create --file fixed-wing-env.yml
2. Using requirements.txt
   Use or activate desired environment
   pip install -r requirements-fixed-wing.txt

# Installation

First install all related code which is under the "magpie/libs" folder

1. fixed-wing-gym
   cd magpie/libs/fixed-wing-gym/
   pip install -e .

2. pyfly
   cd magpie/libs/pyfly/
   pip install -e .
