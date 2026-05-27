"""chimerax-vampnet — VAMPnet / Markov state modeling for protein
conformational ensembles, integrated into the ChimeraX visualization
environment.
"""

__version__ = "0.1.0.dev0"

from chimerax.core.toolshed import BundleAPI


class _VAMPnetBundleAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        # Lazy import so the bundle can be discovered even when optional
        # dependencies (torch, deeptime) aren't yet importable.
        from . import cmd
        cmd.register_commands(logger)


bundle_api = _VAMPnetBundleAPI()
