"""Set up gin.configurables for environment creation."""
from training.wrappers import EpisodeInfo, VecObsNormWrapper
from dl.rl.envs import ActionNormWrapper
import gin
from sofa_envs.sofa_env import SofaEnv

from sofa_envs.termination_fns import get_termination_fn
from sofa_envs.reward_fns import get_reward_fn
from sofa_envs.design_spaces import get_design_space
from coopt.coopt_vec_env import CoOptSubprocVecEnv
from coopt.design_manager import DesignManager


@gin.configurable(blacklist=['nenv'])
def sofa_make_env(scene_id,
                  design_space,
                  reward_fn,
                  termination_fn,
                  dt=0.01,
                  friction_coef=0.07,
                  gravity=[0.0, 0.0, 0.0],
                  max_steps=1000,
                  steps_per_action=30,
                  debug=False,
                  with_gui=False,
                  nenv=1,
                  seed=0,
                  norm_observations=True,
                  norm_actions=True,
                  reduction=None,
                  design_manager=True,
                  default_design=None,
                  **design_params):
    """Create a sofa environment."""

    reward_fn = get_reward_fn(reward_fn)
    termination_fn = get_termination_fn(termination_fn)
    design_space = get_design_space(design_space)(**design_params)

    def _env(rank):
        def _thunk():
            env = SofaEnv(
                scene_id=scene_id,
                design_space=design_space,
                reward_fn=reward_fn,
                termination_fn=termination_fn,
                dt=dt,
                friction_coef=friction_coef,
                max_steps=max_steps,
                gravity=gravity,
                steps_per_action=steps_per_action,
                debug=debug,
                with_gui=with_gui,
                reduction=reduction,
                default_design=default_design,
            )
            if norm_actions:
                env = ActionNormWrapper(env)
            env = EpisodeInfo(env)
            env.seed(seed + rank)
            return env
        return _thunk

    env = CoOptSubprocVecEnv([_env(i) for i in range(nenv)], context='fork')

    if norm_observations:
        env = VecObsNormWrapper(env)
    if design_manager:
        env = DesignManager(env)
    return env
