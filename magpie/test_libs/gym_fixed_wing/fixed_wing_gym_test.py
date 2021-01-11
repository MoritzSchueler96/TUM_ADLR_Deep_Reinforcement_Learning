# imports
import gym
from pyfly.pid_controller import PIDController
import os
from gym_fixed_wing.fixed_wing import FixedWingAircraft
from pyfly_fixed_wing_visualizer.pyfly_fixed_wing_visualizer import simrecorder

startingDir = os.path.dirname(os.path.realpath(__file__))
print(startingDir)
os.chdir(startingDir)
os.chdir("../../libs/fixed-wing-gym/gym_fixed_wing/")

rec = simrecorder(1000)
# create environment from config file
env = FixedWingAircraft(
    "fixed_wing_config.json",
    config_kw={
        "steps_max": 1000,
        "observation": {"noise": {"mean": 0, "var": 0}},
        "action": {"scale_space": False},
    },
    sim_config_kw={
        "recorder": rec,
    },
)
# set seed to be able to repeat results
env.seed(2)
# decide whether tasks should get harder over time
env.set_curriculum_level(1)
# reset environment
obs = env.reset()

# create PID controller
pid = PIDController(env.simulator.dt)
done = False

# do as long as task is not fulfilled
while not done:
    # do one control step
    pid.set_reference(env.target["roll"], env.target["pitch"], env.target["Va"])
    # get phi, theta and va
    phi = env.simulator.state["roll"].value
    theta = env.simulator.state["pitch"].value
    Va = env.simulator.state["Va"].value
    # get state vector
    omega = env.simulator.get_states_vector(["omega_p", "omega_q", "omega_r"])

    # get action
    action = pid.get_action(phi, theta, Va, omega)
    # do simulation step and get reward and observation vector
    obs, rew, done, info = env.step(action)

env.render_on_reset = True
# plot result
env.render(block=True)
print("yeah")
