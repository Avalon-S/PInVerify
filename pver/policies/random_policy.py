import random

class RandomPolicy:
    def __init__(self, cfg, client=None):
        self.cfg = cfg
        # Client not needed for random, but kept for signature compatibility
        
        self.single_view = (cfg.env.max_steps == 1)
        
    def reset(self, obs):
        pass

    def act(self, obs):
        # Action space
        # Decision: Yes, No, Unsure (if multi-view)
        
        choices = ["Yes", "No"]
        if not self.single_view:
             choices.append("Unsure")
             
        decision = random.choice(choices)
        
        action = {"decision": decision}
        
        if decision == "Unsure":
            # Pick random nav direction
            # Using Env's implied implicit directions (0-5 or strings)
            # From Env.py: "nav_dir": spaces.Discrete(6)
            # But MLLM policy used strings: "front", "front-left"...
            # Let's verify MLLM output: action["nav_rel"] = chosen (string)
            # Env._resolve_next_view maps strings.
            
            nav_opts = ["front", "front-left", "front-right", "back", "back-left", "back-right"]
            action["nav_rel"] = random.choice(nav_opts)
            
        return action
