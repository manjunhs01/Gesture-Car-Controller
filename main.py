"""Silent hand-gesture controller: ``python main.py`` (no window, no overlay; see README)."""

import sys

from gesture_car.cli import main

if __name__ == "__main__":
    sys.exit(main())
