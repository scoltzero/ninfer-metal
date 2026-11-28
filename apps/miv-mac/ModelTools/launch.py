"""Launch a third-party client using the MIV-scoped integration profile."""
import json
import os
from pathlib import Path
import sys

from omlx.integrations import get_integration, IntegrationContext


def launch(profile):
    config = json.loads(Path(profile).read_text())
    tool = get_integration(config.pop("tool"))
    if tool is None:
        raise ValueError("未知客户端")
    context = IntegrationContext(**config)
    if not tool.is_installed():
        raise SystemExit(tool.install_hint or "请先安装客户端。")
    tool.launch(context)


if __name__ == "__main__":
    launch(sys.argv[1])
