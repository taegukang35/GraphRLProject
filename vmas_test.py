import vmas
from vmas.simulator.scenario import BaseScenario
import time

scenarios = ["waterfall", "navigation"]
scenario = "navigation"

env = vmas.make_env(
        scenario="navigation", # can be scenario name or BaseScenario class
        num_envs=32,
        device="cpu", # "cpu", "cuda"
        continuous_actions=True,
        wrapper=None, 
        max_steps=None, # Defines the horizon. None is infinite horizon.
        seed=None, # Seed of the environment
        dict_spaces=True, # By default tuple spaces are used with each element in the tuple being an agent.
        # If dict_spaces=True, the spaces will become Dict with each key being the agent's name
        grad_enabled=False, # If grad_enabled the simulator is differentiable and gradients can flow from output to input
        terminated_truncated=False, # If terminated_truncated the simulator will return separate `terminated` and `truncated` flags in the `done()`, `step()`, and `get_from_scenario()` functions instead of a single `done` flag
        n_agents=8, # Additional arguments you want to pass to the scenario initialization
    )

frame_list = []  # For creating a gif
MAX_STEP = 100
RENDER = True

for step in range(MAX_STEP):
    print(f"Step {step}")
    actions = []
    for i, agent in enumerate(env.agents):
        action = env.get_random_action(agent)
        actions.append(action)
    obs, rews, dones, info = env.step(actions)
    if RENDER:
        frame = env.render(mode="rgb_array")
        frame_list.append(frame)

if RENDER:
    from moviepy.editor import ImageSequenceClip
    fps=30
    clip = ImageSequenceClip(frame_list, fps=fps)
    clip.write_gif(f'{scenario}.gif', fps=fps)