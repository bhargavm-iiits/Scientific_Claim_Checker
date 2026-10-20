"""Scientific Claim Checker: retrieve, read and judge scientific claims against SciFact abstracts.

The public entry points are:

* :class:`claim_checker.runtime.ClaimChecker` - shared pipeline used by the CLI, UI and notebook.
* :func:`claim_checker.runtime.check_readiness` - cheap readiness check that loads no models.
* :class:`claim_checker.schemas.ClaimResult` - the validated result record.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
