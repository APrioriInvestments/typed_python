from dataclasses import dataclass

test_input = """A Y
B X
C Z
"""


class Hand:
    def __init__(self, input_str: str):
        match input_str:
            case "A" | "X":
                self.value = 1  # Rock
            case "B" | "Y":
                self.value = 2  # Paper
            case "C" | "Z":
                self.value = 3  # Scissors
            case _:
                raise ValueError()

    def __eq__(self, other):
        return self.value == other.value

    def __gt__(self, other):
        match self.value:
            case 1:
                return other.value == 3
            case 2:
                return other.value == 1
            case 3:
                return other.value == 2

    def __repr__(self):
        return f"Hand({self.value})"


def score_hands(my_hand: Hand, opponent_hand: Hand) -> int:
    score = 0
    if my_hand == opponent_hand:  # a draw
        score = 3
    elif my_hand > opponent_hand:  # a win
        score = 6
    return score + my_hand.value


def parse_input(input_str) -> list[list[Hand]]:
    games = []
    for line in input_str.split("\n"):
        line = line.strip()
        if line:
            opponent_hand, my_hand = line.split()
            games.append([Hand(opponent_hand), Hand(my_hand)])
    return games


def part_one(input_str: str) -> int:
    """
    parse the raw text into a list, then score the strategy.
    """
    games = parse_input(input_str)

    score = sum(score_hands(m, o) for m, o in games)

    return score


assert part_one(test_input) == 15
