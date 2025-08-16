import argparse
from typing import Any, Callable

from sims import play_breakout

SIMS_MAP = {
    "breakout": play_breakout
}

def __validate_game(arg: Any) -> Callable:
    if not isinstance(arg, str) or arg not in SIMS_MAP:
        raise argparse.ArgumentTypeError("Invalid `game` argument")
    
    return SIMS_MAP[arg]


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(prog="RL Playground")
    argparser.add_argument("game", type=__validate_game, help="Specify which game to play")

    args = argparser.parse_args()
    args.game()
