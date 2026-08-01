"""Allow ``python -m mas_cc.cli`` to mirror the console script."""

from .main import main

raise SystemExit(main())
