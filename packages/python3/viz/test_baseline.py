from make_env import sofa_make_env
import numpy as np


def sine_wave_gait(env):

    offset = 110
    A = 45
    P = 4
    dt = 0.33
    def _get_action(t):
        t_start = offset / 360 * P
        action = [
            A - A * np.cos(t * 2 * np.pi / P),
            (t >= t_start) * (A - A * np.cos((t - t_start) * 2 * np.pi / P)),
            0,
        ]
        return np.array([action])

    t = 0
    while t <= 20:
        for _ in range(int(dt * 100)):
            env.step(_get_action(t))
            t += 0.01
    for _ in range(100):
        env.step(np.zeros(3))


env = sofa_make_env('EmptyScene',
                    'DiskWithLegsOpenLoopDesignSpace',
                    'forward_distance',
                    'stop_on_nan',
                    dt=0.01,
                    friction_coef=1.2,
                    gravity=[0.0, 0.0, -9800.0],
                    max_steps=2200,
                    steps_per_action=1,
                    with_gui=True,
                    nenv=1,
                    norm_observations=False,
                    norm_actions=False,
                    design_manager=False,
                    reduction='09-29_tolm_0.0032_tolg_0.0010',
                    default_design=[1, 0, 0, 2, 2, 0, 0, 1],
                    ym=1160,
                    paper_ym=2320)


env.reset()
sine_wave_gait(env)
env.close()
