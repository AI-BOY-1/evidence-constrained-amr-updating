"""Configurable I/O for study-script copies; never overwrite original source."""
import os
from pathlib import Path
RESEARCH_ROOT=Path(__file__).resolve().parent
DATA_ROOT=Path(os.environ.get('NCS_DATA_ROOT',str(RESEARCH_ROOT.parent/'_runtime'))).expanduser().resolve()
def data_path(relative=''):
    return DATA_ROOT/relative
