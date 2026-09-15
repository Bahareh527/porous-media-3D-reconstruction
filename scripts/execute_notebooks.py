"""Execute maintained notebooks in separate temporary kernels without saving outputs."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    with TemporaryDirectory(prefix="porous-kernels-") as temporary:
        kernel = Path(temporary) / "validation"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Repository validation",
                    "language": "python",
                    "env": {
                        "IPYTHONDIR": str(Path(temporary) / "ipython"),
                        "TF_CPP_MIN_LOG_LEVEL": "2",
                    },
                }
            )
        )
        specification = KernelSpecManager(kernel_dirs=[temporary])
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            manager = KernelManager(
                kernel_name="validation",
                kernel_spec_manager=specification,
                connection_file=str(Path(temporary) / f"{path.stem}.json"),
            )
            NotebookClient(
                notebook, km=manager, timeout=300, resources={"metadata": {"path": str(ROOT)}}
            ).execute(cleanup_kc=True)
            print(f"PASS {path.name}", flush=True)


if __name__ == "__main__":
    main()
