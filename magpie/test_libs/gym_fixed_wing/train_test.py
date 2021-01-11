from gym_fixed_wing.examples import train_rl_controller
import os

startingDir = os.path.dirname(os.path.realpath(__file__))
print(startingDir)
os.chdir(startingDir)
os.chdir("../../libs/fixed-wing-gym/gym_fixed_wing/examples/")

if __name__ == "__main__":

    train_rl_controller.main(
        model_name="reproduceMLP",
        num_envs=4,
        disable_curriculum=True,
        policy="MLP",
        train_steps=5000000,
        test_data_path="test_sets/test_set_wind_none_step20-20-3.npy",
    )
