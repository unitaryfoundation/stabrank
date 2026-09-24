"""Run one probe under a wall-clock cap at nice 19, in its own session, so
that the cap kills the whole process group (macOS has no `timeout`).

    run.py [--max-seconds 600] [--log PATH] -- CMD ARGS...

The command is run through `uv run --extra challenge python` when it is a
.py path; anything else is run as given. Exit status: the child's, or 124
when the cap killed it.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def kill_group(proc):
    """SIGKILL the child's process group; when macOS refuses the group
    signal (EPERM, seen once here, after which the child ran on past the
    cap), kill the group's members one by one, then the child."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        return
    except ProcessLookupError:
        return
    except PermissionError:
        pass
    try:
        out = subprocess.run(["ps", "-o", "pid=", "-g", str(proc.pid)], capture_output=True, text=True).stdout
        pids = [int(x) for x in out.split()]
    except (OSError, ValueError):
        pids = []
    for pid in pids + [proc.pid]:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-seconds", type=float, default=600.0)
    ap.add_argument("--log", default=None, help="append stdout and stderr of the child here as well")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv[1:])
    cmd = a.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command")
    if cmd[0].endswith(".py"):
        cmd = ["uv", "run", "--extra", "challenge", "python"] + cmd
    cmd = ["nice", "-n", "19"] + cmd
    log = open(a.log, "a") if a.log else None
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    killed = False
    try:
        while True:
            line = proc.stdout.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
                if log:
                    log.write(line)
                    log.flush()
            elif proc.poll() is not None:
                break
            if time.time() - t0 > a.max_seconds and not killed:
                killed = True
                kill_group(proc)
    finally:
        kill_group(proc)
        proc.wait()
    dt = time.time() - t0
    msg = f"[run.py] {'KILLED at cap' if killed else 'exit ' + str(proc.returncode)} after {dt:.1f}s\n"
    sys.stdout.write(msg)
    if log:
        log.write(msg)
        log.close()
    return 124 if killed else proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv))
