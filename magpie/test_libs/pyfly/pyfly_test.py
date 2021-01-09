from pyfly.pyfly import PyFly
from pyfly.pid_controller import PIDController
import numpy as np
import os

# set dir for config and parameter file
startingDir = os.path.dirname(os.path.realpath(__file__))
print(startingDir)
os.chdir(startingDir)
os.chdir("../../libs/pyfly/pyfly/")

# load object with config and param file
sim = PyFly("pyfly_config.json", "x8_param.mat")
sim.reset(state={"roll": -0.5, "pitch": 0.15})

# set seed for repeatability
sim.seed(0)

# init pid controller
pid = PIDController(sim.dt)

# set start vector
pid.set_reference(phi=0.2, theta=0, va=22)

# simulate for 500 timesteps
for step_i in range(500):
    # get state
    phi = sim.state["roll"].value
    theta = sim.state["pitch"].value
    Va = sim.state["Va"].value
    omega = [
        sim.state["omega_p"].value,
        sim.state["omega_q"].value,
        sim.state["omega_r"].value,
    ]

    # calculate action based on control deviation
    action = pid.get_action(phi, theta, Va, omega)
    # simulate one timestep
    success, step_info = sim.step(action)

    if not success:
        break

# render plots of simulation as specified in the config
sim.render(block=True)
