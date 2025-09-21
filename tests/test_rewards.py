import pytest
import verifiers as vf
from rewards import check_answer_reward_func


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("arise", 0.0),
        ("slept", 1.0),
    ],
)
def test_check_answer_reward_func_with_real_completion(answer, expected):
    parser = vf.XMLParser(
        fields=["think", "guess", "advice"], answer_field="guess")
    completion = [{'content': "<think>Since this is the first guess, I should choose a common 5-letter word with frequently used letters to maximize the information gained about the secret word.</think> \n<guess>[arise]</guess> \n<advice>Start with a common word that includes vowels and frequently used consonants to gather as much information as possible about the secret word's letters and their positions.</advice>",
                   'role': 'assistant'},
                  {'content': '\nC R A N E\nX X X X Y\nYou have 5 guesses left.',
                   'role': 'user'},
                  {'content': "<think>The feedback shows that all letters except 'E' are not in the word, and 'E' is in the word but in a different position than the last letter. I should now try a word with an 'E' but not at the end, and avoid using the letters C, R, A, N since they are marked X. I'll try a word with new letters including 'E' somewhere in the middle or beginning.</think> \n<guess>[spelt]</guess> \n<advice>Since 'E' is in the word but not at the end, try placing 'E' in different positions. Avoid letters that were marked as not in the word (C, R, A, N, and also E at the end), and test new common letters to narrow down possibilities.</advice>",
                   'role': 'assistant'},
                  {'content': "\nC R A N E\nX X X X Y\nYou have 5 guesses left.\n[GAME] [SLEDY]\n[GAME] You attempted an invalid move. Reason: 'sledy' is not an English word. Please resubmit a valid move and remember to follow the game rules to avoid penalties.",
                   'role': 'user'},
                  {'content': "<think>The previous guess was invalid because 'sledy' is not a valid English word. I need to choose a proper English word with 'E' in it, but not at the end, and avoid using letters C, R, A, N. I will pick a common word that fits these conditions, for example, 'spite', but 'E' is at the end there. Let's try 'spike' with 'E' at the end again; no, 'E' can't be at the end. How about 'spine'—'E' at the end again. 'Spine' is invalid for 'E' placement. Maybe 'style'—'E' at the end again. Let's try 'elope'—has 'E' at the end. Can't do that. Let's try 'spelt' or 'slept'—'E' is second or third letter. 'Slept' is valid. Let's try 'slept'.</think> \n<guess>[slept]</guess> \n<advice>Make sure your guess is a valid English word. Keep 'E' in the word but not at the end, and avoid letters that the previous guess ruled out. Try placing 'E' in different positions and use new letters to gain more clues.</advice>",
                   'role': 'assistant'},
                  {'content': "\nC R A N E\nX X X X Y\nYou have 5 guesses left.\n[GAME] [SLEDY]\n[GAME] You attempted an invalid move. Reason: 'sledy' is not an English word. Please resubmit a valid move and remember to follow the game rules to avoid penalties.\n[GAME] [BELIT]",
                   'role': 'user'}]

    reward = check_answer_reward_func(parser, completion, answer)
    assert reward == expected
