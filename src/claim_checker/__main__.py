"""Allow ``python -m claim_checker <command>`` (same as the ``claim-checker`` script)."""

from claim_checker.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
