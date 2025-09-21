# Wordle Tutor

### Overview
- **Environment ID**: `wordle-tutor`
- **Short description**: A tutor to teach you Wordle
- **Tags**: multi-turn, game

### Datasets
- **Primary dataset(s)**: <name(s) and brief description>
- **Source links**: <links>
- **Split sizes**: <train/eval counts>

### Task
- **Type**: multi-turn
- **Parser**: XMLParser
- **Rubric overview**: solving the puzzle, with partial credit and format rewards

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval wordle-tutor   -m gpt-4.1-mini   -n 1 -r 3 -t 1024 -T 0.7   -a '{"num_train_examples": 1, "num_eval_examples": 1, "guesser": {"model": "gpt-4.1-mini", "max_output_tokens": 16} }'

uv run vf-eval wordle-tutor   -m gpt-4.1-mini   -n 1 -r 3 -t 1024 -T 0.7   -a '{"num_train_examples": 1, "num_eval_examples": 1, "guesser": {"model": "gpt-5", "max_output_tokens": 100, "reasoning": {"effort": "minimal"} } }'
```

Configure model and sampling:

```bash
uv run vf-eval wordle-tutor   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | 1000 | Number of Training Examples |
| `num_eval_examples` | int | 1000 | Number of Eval Examples |
| `guesser.model` | str |  | Model to simulate user |
| `guesser.max_output_tokens` | int |  | Control the output tokens |
| `guesser.reasoning` | dict |  | Reasoning effort |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |

