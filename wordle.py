import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv
from prompts import THINK_GUESS_SYSTEM_PROMPT, NOTHINK_GUESS_SYSTEM_PROMPT
from rewards import (check_answer_reward_func,
                     count_turns_reward_func, partial_credit_reward_func)
from feedback import wordle_feedback_fn


class WordleEnv(TextArenaEnv):

    def __init__(
        self,
        num_train_examples: int = 2000,
        num_eval_examples: int = 20,
        use_think: bool = True,
        **kwargs,
    ):
        if use_think:
            system_prompt = THINK_GUESS_SYSTEM_PROMPT
            parser = vf.XMLParser(
                fields=["think", "guess"], answer_field="guess")
        else:
            system_prompt = NOTHINK_GUESS_SYSTEM_PROMPT
            parser = vf.XMLParser(fields=["guess"], answer_field="guess")

        rubric = vf.Rubric(parser=parser)
        rubric.add_reward_func(check_answer_reward_func)
        rubric.add_reward_func(partial_credit_reward_func)
        rubric.add_reward_func(count_turns_reward_func)
        rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)

        super().__init__(
            game="Wordle-v0",
            num_train_examples=num_train_examples,
            num_eval_examples=num_eval_examples,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            feedback_fn=wordle_feedback_fn,
            **kwargs,
        )


def load_environment(
    num_train_examples: int = 2000,
    num_eval_examples: int = 20,
    use_think: bool = True,
):

    return WordleEnv(
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        use_think=use_think,
    )
