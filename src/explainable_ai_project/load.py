from pathlib import Path

import pandas as pd
import requests

# path handling
this_file = Path(__file__).resolve()
package_folder = this_file.parent
src_folder = package_folder.parent
project_root = src_folder.parent

raw_datasets_folder = project_root / "datasets" / "root"

