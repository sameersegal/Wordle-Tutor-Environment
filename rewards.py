# correctness
def check_answer_reward_func(parser, completion, answer, **kwargs) -> float:
    message = parser.get_assistant_messages(completion)[-1]
    result = parser.parse(message["content"])
    return 1.0 if result.guess == "[" + answer + "]" else 0.0


def check_user_answer_reward_func(parser, completion, answer, state, **kwargs) -> float:
    return 1.0 if state["is_finished"] else 0.0

# efficiency


def count_turns_reward_func(parser, completion, answer, state, **kwargs) -> float:
    num_turns = len([x for x in completion if x["role"] == "assistant"])
    is_correct = check_user_answer_reward_func(
        parser, completion, answer, state, **kwargs)
    return is_correct / (num_turns + 1)

# partial credit


def partial_credit_reward_func(parser, completion, answer, state, **kwargs) -> float:
    """Reward function that gives partial credit for the correct guess."""

    is_completed = check_user_answer_reward_func(
        parser, completion, answer, state, **kwargs)
    if is_completed:
        return 1.0

    user_messages = [message["content"].strip()
                     for message in parser.get_user_messages(completion)]
    letters = {}
    # Precedence of statuses
    rank = {'X': 0, 'Y': 1, 'G': 2}

    for user_message in user_messages:
        guess, scoring = user_message.split("\n")[:2]
        for g, s in zip(guess.split(), scoring.split()):
            # keep the best (max) status seen so far for each letter
            letters[g] = max(letters.get(g, 'X'), s, key=rank.get)

    num_greens = sum(s == "G" for s in letters.values())
    num_yellows = sum(s == "Y" for s in letters.values())

    return 0.2 * num_greens + 0.1 * num_yellows
