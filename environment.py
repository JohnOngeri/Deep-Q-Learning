import gymnasium
import ale_py

gymnasium.envs.registration.register_envs(ale_py)
print("✅ ALE environments registered successfully!")

# Test the environment
env = gymnasium.make("ALE/Breakout-v5", render_mode="rgb_array")
obs, info = env.reset()
print("✅ Atari environment loaded successfully!")
env.close()
