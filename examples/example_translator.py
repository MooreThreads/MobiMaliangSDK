
import os
import sys
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))
from modules.wrappers import Translator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the translator model.",
    )

    parser.add_argument(
        "--chinese_text",
        default=None,
        type=str,
        required=True,
        help="Chinese sentence(s) or phrase(s) or word(s)",
    )

    args = parser.parse_args()

    tagger = Translator(device="cpu")
    tagger.load_models(model_path=args.checkpoint_path)
    english = tagger(args.chinese_text)
    print(english)
