"""``python -m sensortrust`` entry point (GUI by default, see ``--help``)."""

import multiprocessing

from .cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()  # required by the frozen Windows executable
    main()
