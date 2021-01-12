from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import SAC
from stable_baselines3 import mSAC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecCheckNan,
    DummyVecEnv,
    VecNormalize,
)
from gym_fixed_wing.fixed_wing import FixedWingAircraft
from pyfly.pid_controller import PIDController
from pyfly_fixed_wing_visualizer.pyfly_fixed_wing_visualizer import simrecorder
import os
import numpy as np
import sys


def make_env(config_path, rank, seed=0, info_kw=(), config_kw=None, sim_config_kw=None):
    """
    Utility function for multiprocessed env.

    :param config_path: (str) path to gym environment configuration file
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    :param info_kw: ([str]) list of entries in info dictionary that the Monitor should extract
    :param config_kw: (dict) dictionary of key value pairs to override settings in the configuration file of the gym environment
    :param sim_config_kw (dict) dictionary of key value pairs to override settings in the configuration file of PyFly
    """

    def _init():
        env = FixedWingAircraft(
            config_path, config_kw=config_kw, sim_config_kw=sim_config_kw
        )
        env = Monitor(
            env, filename=None, allow_early_resets=True, info_keywords=info_kw
        )
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def print_results(results):
    for metric, _ in results.items():
        if isinstance(results[metric], dict):
            print(metric)
            for state, v in results[metric].items():
                if metric != "success":
                    v = np.nanmean(
                        [
                            val if results["success"]["all"][i] else np.nan
                            for i, val in enumerate(v)
                        ]
                    )
                    print("\t{}:\t{}".format(state, v))
                else:
                    print("\t{}:\t{}".format(state, np.nanmean(v)))


def evaluate_model_on_set(
    set_path,
    model,
    config_path=None,
    config_kw=None,
    metrics=("success", "control_variation", "rise_time", "overshoot", "settling_time"),
    norm_data_path=None,
    num_envs=1,
    turbulence_intensity="None",
    use_pid=False,
    writer=None,
    timestep=None,
):
    """
    :param set_path: (str) path to test set file
    :param model: (PPO2 object or [PIDController]) the controller to be evaluated
    :param config_path: (str) path to gym environment configuration file
    :param config_kw: (dict) dictionary of key value pairs to override settings in the configuration file of the gym environment
    :param metrics: ([str]) list of metrics to be computed and recorded
    :param norm_data_path: (str) path to folder containing normalization statistics
    :param num_envs: (int) number of gym environments to run in parallell using multiprocessing
    :param turbulence_intensity: (str) the intensity setting of the wind turbulence
    :param use_pid: (bool) Whether the evaluated controller is a PID controller or not
    :param writer: (tensorboard writer) If supplied, evaluation results will be written to tensorboard log, if not, results are printed to standard output
    :param timestep: (int) What timestep results are written to when using tensorboard logging
    :return: (dict) the metrics computed for the evaluated controller on the test set
    """
    scenarios = list(np.load(set_path, allow_pickle=True))
    scenario_count = len(scenarios)

    if config_kw is None:
        config_kw = {}

    config_kw.update(
        {
            "steps_max": 1500,
            "target": {
                "on_success": "done",
                "success_streak_fraction": 1,
                "success_streak_req": 100,
                "states": {0: {"bound": 5}, 1: {"bound": 5}, 2: {"bound": 2}},
            },
        }
    )

    if use_pid:
        config_kw["action"] = {"scale_space": False}

    sim_config_kw = {
        "turbulence": turbulence_intensity != "None",
        "turbulence_intensity": turbulence_intensity,
    }

    # sim_config_kw.update({"recorder": simrecorder(1500)})

    test_env = SubprocVecEnv(
        [
            make_env(config_path, i, config_kw=config_kw, sim_config_kw=sim_config_kw)
            for i in range(num_envs)
        ]
    )

    if use_pid:
        dt = test_env.get_attr("simulator")[0].dt
        for pid in model:
            pid.dt = dt
        env_cfg = test_env.get_attr("cfg")[0]
        obs_states = [var["name"] for var in env_cfg["observation"]["states"]]
        try:
            phi_i, theta_i, Va_i = (
                obs_states.index("roll"),
                obs_states.index("pitch"),
                obs_states.index("Va"),
            )
            omega_i = [
                obs_states.index("omega_p"),
                obs_states.index("omega_q"),
                obs_states.index("omega_r"),
            ]
        except ValueError:
            print(
                "When using PID roll, pitch, Va, omega_p, omega_q, omega_r must be part of the observation vector."
            )
    else:
        test_env = VecNormalize(test_env)
        if model.env is not None:
            test_env.obs_rms = model.env.obs_rms
            test_env.ret_rms = model.env.ret_rms
        else:
            assert norm_data_path is not None
            model.env = test_env.load(os.path.join(norm_data_path, "env.pkl"), test_env)
        test_env.training = False

    res = {metric: {} for metric in metrics}
    res["rewards"] = [[] for i in range(scenario_count)]
    active_envs = [i < scenario_count for i in range(num_envs)]
    env_scen_i = [i for i in range(num_envs)]
    test_done = False
    obs = np.array(
        [np.zeros(test_env.observation_space.shape) for i in range(num_envs)]
    )
    done = [True for i in range(num_envs)]
    info = None

    while not test_done:
        for i, env_done in enumerate(done):
            if env_done:
                if len(scenarios) > 0 or active_envs[i]:
                    if len(scenarios) > 0:
                        print(
                            "{}/{} scenarios left".format(
                                len(scenarios), scenario_count
                            )
                        )
                        scenario = scenarios.pop(0)
                        env_scen_i[i] = (scenario_count - 1) - len(scenarios)
                        # test_env.env_method("render", indices=0, mode="plot", show=True, close=True)
                        obs[i] = test_env.env_method("reset", indices=i, **scenario)[0]
                        if use_pid:
                            model[i].reset()
                            model[i].set_reference(
                                scenario["target"]["roll"],
                                scenario["target"]["pitch"],
                                scenario["target"]["Va"],
                            )
                    else:
                        active_envs[i] = False
                    if info is not None:
                        for metric in metrics:
                            if isinstance(info[i][metric], dict):
                                for state, value in info[i][metric].items():
                                    if state not in res[metric]:
                                        res[metric][state] = []
                                    res[metric][state].append(value)
                            else:
                                if "all" not in res[metric]:
                                    res[metric]["all"] = []
                                res[metric]["all"].append(info[i][metric])

        if len(scenarios) == 0:
            test_done = not any(active_envs)
        if use_pid:
            actions = []
            for i, pid in enumerate(model):
                roll, pitch, Va = obs[i, phi_i], obs[i, theta_i], obs[i, Va_i]
                omega = obs[i, omega_i]
                if info is not None and "target" in info[i]:
                    pid.set_reference(
                        phi=info[i]["target"]["roll"],
                        theta=info[i]["target"]["pitch"],
                        va=info[i]["target"]["Va"],
                    )
                actions.append(pid.get_action(roll, pitch, Va, omega))
            actions = np.array(actions)
        else:
            actions, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = test_env.step(actions)
        for i, env_rew in enumerate(rew):
            res["rewards"][env_scen_i[i]].append(env_rew)

    if writer is not None:
        for metric, metric_v in res.items():
            if isinstance(res[metric], dict):
                for state, v in res[metric].items():
                    writer.record(
                        key="test_set/{}_{}".format(metric, state),
                        value=np.nanmean(v),
                    )
    else:
        print_results(res)

        return res


def main(
    path_to_file,
    num_envs=None,
    model_path=None,
    env_config_path=None,
    use_pid=False,
    turbulence_intensity="None",
    print_res=False,
):

    if print_res:
        print_results(np.load(path_to_file, allow_pickle=True).item())
        sys.exit(0)

    if num_envs is not None:
        num_cpu = int(num_envs)
    else:
        num_cpu = 1

    if use_pid:
        assert env_config_path is not None
        model = [PIDController() for i in range(num_cpu)]
        config_path = env_config_path
        norm_data_path = None
    else:
        assert model_path is not None
        model_path = model_path
        if "mSAC" in model_path:
            model = mSAC.load(path=model_path)
        elif "SAC" in model_path:
            model = SAC.load(path=model_path)
        else:
            model = PPO.load(path=model_path)

        config_path = os.path.join(
            os.path.dirname(model_path), "fixed_wing_config.json"
        )
        norm_data_path = os.path.dirname(model_path)

    np.save(
        "eval_res.npy",
        evaluate_model_on_set(
            path_to_file,
            model,
            config_path=config_path,
            num_envs=num_cpu,
            use_pid=use_pid,
            norm_data_path=norm_data_path,
            turbulence_intensity=turbulence_intensity,
        ),
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "path_to_file",
        help="Path to test set to evaluate, or if --print-results flag is set, the path to the results file.",
    )
    parser.add_argument(
        "--num-envs", required=False, help="Number of processes for environments."
    )
    parser.add_argument(
        "--model-path", required=False, help="Path to RL controller model file."
    )
    parser.add_argument(
        "--env-config-path",
        required=False,
        help="Path to environment configuration file for PID controller.",
    )
    parser.add_argument(
        "--PID",
        dest="use_pid",
        action="store_true",
        required=False,
        help="Use PID controller",
    )
    parser.add_argument(
        "--turbulence-intensity",
        required=False,
        help="Intensity of turbulence, one of [none, light, moderate, severe]",
    )
    parser.add_argument(
        "--print-results",
        required=False,
        action="store_true",
        help="Print results from file at path_to_file",
    )

    args = parser.parse_args()

    main(
        path_to_file=args.path_to_file,
        num_envs=args.num_envs,
        model_path=args.model_path,
        env_config_path=args.env_config_path,
        use_pid=args.use_pid,
        turbulence_intensity=args.turbulence_intensity,
        print_res=args.print_results,
    )
