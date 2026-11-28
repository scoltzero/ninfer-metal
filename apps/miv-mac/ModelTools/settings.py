"""Settings surface consumed by the private model-tools snapshot."""
import os
import subprocess
from types import SimpleNamespace

def get_settings():
    return SimpleNamespace(
        huggingface=SimpleNamespace(endpoint=os.environ.get("HF_ENDPOINT", "")),
        modelscope=SimpleNamespace(endpoint=os.environ.get("MODELSCOPE_DOMAIN", "")),
    )

def get_system_memory():
    return int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"]))
