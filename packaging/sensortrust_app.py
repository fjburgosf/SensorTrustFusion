"""Entry point of the frozen Windows executable (PyInstaller).

Without arguments the graphical interface opens; with arguments the command
line interface is used (``SensorTrustFusion.exe benchmark ST-BENCH-02``).
"""

import multiprocessing
import os
import sys


def _console_streams():
    """A windowed executable has no console: send output to a log file in the user folder."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    stream = None
    for folder in (os.path.join(os.path.expanduser("~"), "Documents", "SensorTrustFusion"),
                   os.path.join(os.path.expanduser("~"), "SensorTrustFusion"), os.getcwd()):
        try:
            os.makedirs(folder, exist_ok=True)
            stream = open(os.path.join(folder, "sensortrust_cli.log"), "a", encoding="utf-8")
            break
        except OSError:
            continue
    if stream is None:
        stream = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream  # required for Monte Carlo worker processes in a frozen app


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _console_streams()
    if len(sys.argv) > 1:
        from sensortrust.gui.app import _close_splash
        _close_splash()
        from sensortrust.cli import main
        main(sys.argv[1:])
    else:
        from sensortrust.gui.app import main
        main()
