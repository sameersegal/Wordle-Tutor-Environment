# feedback functions
def wordle_feedback_fn(observation: str) -> str:
    if "Feedback:" in observation:
        return observation.split("Feedback:")[-1]
    else:
        return observation


def wordle_tutor_feedback_fn(guess: str, observation: str) -> str:
    output = f"<guess>[{guess}]</guess>\n"

    if "Feedback:" in observation:
        output += observation.split("Feedback:")[-1]
    else:
        output += observation
    return output
