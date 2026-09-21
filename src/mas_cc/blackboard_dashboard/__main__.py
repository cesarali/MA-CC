"""``python -m mas_cc.blackboard_dashboard ...`` is ``mas-cc blackboard dashboard ...``."""
from __future__ import annotations

import sys

from mas_cc.cli.main import main

if __name__ == "__main__":
    sys.exit(main(["blackboard", "dashboard", *sys.argv[1:]]))
