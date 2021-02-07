# from gym_fixed_wing.examples import train_rl_controller

from train_rl_controller import main
import os

startingDir = os.path.dirname(os.path.realpath(__file__))
print(startingDir)
os.chdir(startingDir)
examplesDir = "../libs/fixed-wing-gym/gym_fixed_wing/examples/"
print(examplesDir)

import numpy as np

res = list(
    np.load(
        examplesDir + "test_sets/test_set_wind_none_step20-20-3.npy", allow_pickle=True
    )
)[0]
print(res)

if __name__ == "__main__":

    main(
        model_name="mSAC__test1___",
        num_envs=4,
        env_config_path=startingDir,
        disable_curriculum=True,
        policy="MlpPolicy",
        train_steps=3000000,
        test_data_path=examplesDir + "test_sets/test_set_wind_none_step20-20-3.npy",
    )
