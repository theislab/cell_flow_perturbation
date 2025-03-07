from importlib import metadata

from cfp import data, datasets, external, metrics, model, networks, solvers, training, utils
from cfp import preprocessing as pp

__version__ = metadata.version("cell_flow_perturbation")
try:
    md = metadata.metadata(__name__)
    __version__ = md.get("version", "")  # type: ignore[attr-defined]
    __author__ = md.get("Author", "")  # type: ignore[attr-defined]
    __maintainer__ = md.get("Maintainer-email", "")  # type: ignore[attr-defined]
except ImportError:
    md = None
    __author__ = "TODO"

del metadata, md
