from gym_fixed_wing.examples import train_rl_controller
import os

startingDir = os.path.dirname(os.path.realpath(__file__))
print(startingDir)
os.chdir(startingDir)
os.chdir("../papers/fixed-wing-gym/gym_fixed_wing/examples/")

if __name__ == "__main__":

    train_rl_controller.main(
        model_name="reproduceMLP",
        num_envs=4,
        disable_curriculum=True,
        policy="MLP",
        train_steps=3000000,
        test_path="test_sets/",
    )
