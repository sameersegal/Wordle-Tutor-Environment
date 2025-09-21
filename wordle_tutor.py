import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv

# prompts
THINK_GUESS_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step inside <think>...</think> tags, \
then follow the instructions inside <guess>...</guess> tags."""

NOTHINK_GUESS_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, give only your guess inside <guess>...</guess> tags."""


# feedback functions
def wordle_feedback_fn(observation: str) -> str:
    if "Feedback:" in observation:
        return observation.split("Feedback:")[-1]
    else:
        return observation


# reward functions
def check_answer_reward_func(parser, completion, answer, **kwargs) -> float:
    guess = parser.parse_answer(completion)
    return 1.0 if guess == "[" + answer + "]" else 0.0


def count_turns_reward_func(parser, completion, answer, **kwargs) -> float:
    num_turns = len([x for x in completion if x["role"] == "assistant"])
    is_correct = check_answer_reward_func(parser, completion, answer, **kwargs)
    return is_correct / (num_turns + 1)


def partial_credit_reward_func(parser, completion, answer, **kwargs) -> float:
    """Reward function that gives partial credit for the correct guess."""

    is_completed = check_answer_reward_func(
        parser, completion, answer, **kwargs)
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
    total = len(letters)

    return 0.2 * num_greens + 0.1 * num_yellows


# environment loader
def load_environment(
    num_train_examples: int = 2000,
    num_eval_examples: int = 20,
    use_think: bool = True,
):
    if use_think:
        system_prompt = THINK_GUESS_SYSTEM_PROMPT
        parser = vf.XMLParser(fields=["think", "guess"], answer_field="guess")
    else:
        system_prompt = NOTHINK_GUESS_SYSTEM_PROMPT
        parser = vf.XMLParser(fields=["guess"], answer_field="guess")

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(check_answer_reward_func)
    rubric.add_reward_func(partial_credit_reward_func)
    rubric.add_reward_func(count_turns_reward_func)
    rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)

    vf_env = TextArenaEnv(
        game="Wordle-v0",
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        feedback_fn=wordle_feedback_fn,
    )
    return vf_env
