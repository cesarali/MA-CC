"""Provider-independent multi-agent language game framework.

The package root deliberately exposes only static metadata.  Importing it is
safe in offline processes: providers, credentials, and observability clients
are loaded only by the modules that use them.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
