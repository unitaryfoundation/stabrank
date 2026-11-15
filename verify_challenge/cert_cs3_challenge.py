"""No-argument adapter for the upstream certificate interface.

Copy beside the eleven unchanged portable_v1 cert_*.py modules. This wrapper
delegates to their bounded full run and never reads a stored result. Its
separate review does not constitute a new execution of the scientific proof.
"""
import sys
import multiprocessing
import cert_cs3_main


if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise SystemExit('This adapter takes no arguments; use cert_cs3_main.py for explicit modes.')
    multiprocessing.freeze_support()
    sys.argv = [sys.argv[0], '--run', '--workers', '4']
    cert_cs3_main.main()
