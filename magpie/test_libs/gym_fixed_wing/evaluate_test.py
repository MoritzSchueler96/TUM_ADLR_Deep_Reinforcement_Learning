from gym_fixed_wing.examples import evaluate_controller
import os

startingDir = os.path.dirname(os.path.realpath(__file__))
print(startingDir)
os.chdir(startingDir)
os.chdir("../../libs/fixed-wing-gym/gym_fixed_wing/examples/")

# needed to avoid that multiprocessing creates recursive subprocesses
if __name__ == "__main__":
    evaluate_controller.main(
        path_to_file="test_sets/test_set_wind_moderate_step20-20-3.npy",
        num_envs=4,
        model_path="models/reproduceMLP/model.pkl",
        turbulence_intensity="moderate",
        print_res=False,
    )
