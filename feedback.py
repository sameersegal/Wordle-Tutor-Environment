# feedback functions
def wordle_feedback_fn(observation: str) -> str:
    if "Feedback:" in observation:
        return observation.split("Feedback:")[-1]
    else:
        return observation
