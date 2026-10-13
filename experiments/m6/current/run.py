#!/usr/bin/env python3
"""Stable entrypoint for scripts/m6/run.sh; always writes a results directory."""
from run_operational_qualification import main
if __name__ == '__main__':
    raise SystemExit(main())
