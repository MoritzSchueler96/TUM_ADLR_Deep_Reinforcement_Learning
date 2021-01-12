# from gym_fixed_wing.examples import evaluate_controller
from evaluate_controller import main
import os

startingDir = os.path.dirname(os.path.realpath(__file__))
print(startingDir)
os.chdir(startingDir)
examplesDir = "../libs/fixed-wing-gym/gym_fixed_wing/examples/"
print(examplesDir)

# needed to avoid that multiprocessing creates recursive subprocesses
if __name__ == "__main__":
    main(
        path_to_file=examplesDir + "test_sets/test_set_wind_moderate_step20-20-3.npy",
        num_envs=1,
        model_path="models/test3/model.pkl",
        turbulence_intensity="moderate",
        print_res=False,
    )
