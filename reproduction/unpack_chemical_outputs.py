"""Extract saved replacement output arrays without recalculating results."""
from pathlib import Path
import argparse
import importlib.util


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("unpack_inputs", Path(__file__).parent / "protein/unpack_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import sys
    sys.argv.extend(["--archive", str(Path(__file__).with_name("chemical_replacement_outputs.tar.gz"))])
    module.main()
