from copy import deepcopy
from textwrap import dedent
from typing import Any
from openai import OpenAI
import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv
from prompts import THINK_GUESS_ADVICE_SYSTEM_PROMPT
from rewards import (
    check_answer_reward_func,
    count_turns_reward_func,
    partial_credit_reward_func,
)
from feedback import wordle_tutor_feedback_fn
from verifiers.types import (
    Messages,
    State,
)


class WordleTutorEnv(TextArenaEnv):

    def __init__(
        self,
        num_train_examples: int = 2000,
        num_eval_examples: int = 20,
        guesser: dict = {"model": "gpt-4.1-mini", "max_output_tokens": 16},
        **kwargs,
    ):
        system_prompt = THINK_GUESS_ADVICE_SYSTEM_PROMPT
        parser = vf.XMLParser(
            fields=["think", "guess", "advice"], answer_field="advice")

        rubric = vf.Rubric(parser=parser)
        rubric.add_reward_func(check_answer_reward_func)
        rubric.add_reward_func(partial_credit_reward_func)
        rubric.add_reward_func(count_turns_reward_func)
        rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)

        self.guesser_client = OpenAI()
        self.guesser_model = guesser.get("model", "gpt-4.1-mini")
        self.guesser_max_output_tokens = guesser.get("max_output_tokens", 16)

        super().__init__(
            game="Wordle-v0",
            num_train_examples=num_train_examples,
            num_eval_examples=num_eval_examples,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            feedback_fn=wordle_tutor_feedback_fn,
            **kwargs,
        )

    async def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> tuple[Messages, State]:
        # load env
        if "ta_env" not in state:
            ta_env = deepcopy(self.ta_env)
            ta_env.reset(num_players=1)
            ta_env.state.game_state["secret_word"] = state["answer"]
            state["ta_env"] = ta_env
        else:
            ta_env = state["ta_env"]

        # parse advice
        assert isinstance(messages[-1], dict)
        advice = self.parser.parse_answer(messages)

        # make a guess
        game_state = ta_env._render_board()
        guess = await self.guess(advice=advice, game_state=game_state)

        # step env
        is_finished, _ = ta_env.step(f"[{guess}]")
        state["is_finished"] = is_finished
        _, observation = ta_env.get_observation()
        feedback = self.feedback_fn(guess, observation)
        return [{"role": "user", "content": str(feedback)}], state

    async def guess(self, advice: str, game_state: str) -> str:

        system_prompt = dedent(f"""You are playing Wordle with the help of an expert tutor. \
        A secret 5-letter word has been chosen. You have 6 attempts to guess it.
        Feedback for each letter will be given as follows:
          - G (green): correct letter in the correct position
          - Y (yellow): letter exists in the word but in the wrong position
          - X (wrong): letter is not in the word

        Game state so far:
        {game_state}
        
        Make your next guess based on the advice from your tutor.
        Respond with only the 5-letter word you are guessing.
        """)

        if advice == "" or advice is None:
            advice = "Start with any common 5-letter word like 'ARISE' or 'PLANT'."

        response = self.guesser_client.responses.create(
            model=self.guesser_model,
            instructions=system_prompt.strip(),
            input=advice.strip(),
            max_output_tokens=self.guesser_max_output_tokens,
        )

        if response.status != "completed" or response.error is not None:
            raise ValueError(
                f"Guesser model failed with status {response.status} and error {response.error}"
            )

        response_text: str = response.output_text.strip()

        return response_text.upper().replace(" ", "")


def load_environment(
    num_train_examples: int = 2000,
    num_eval_examples: int = 20,
    guesser: dict = {"model": "gpt-4.1-mini", "max_output_tokens": 16}
):

    return WordleTutorEnv(
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        guesser=guesser,
    )
