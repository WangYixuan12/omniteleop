"""Check Isaac / OmniGibson compatibility without launching either runtime."""
import ast
import importlib.metadata
import importlib.util
from pathlib import Path


def isaac_version_problem():
    spec = importlib.util.find_spec('omnigibson')
    if spec is None or spec.origin is None:
        return 'omnigibson is not installed in this Python; use the behavior environment'
    try:
        installed = importlib.metadata.version('isaacsim')
    except importlib.metadata.PackageNotFoundError:
        # Standalone Isaac installs can provide the runtime without pip metadata.
        return None
    simulator = Path(spec.origin).parent / 'simulator.py'
    supported = None
    for node in ast.parse(simulator.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'KIT_FILES'
                                               for t in node.targets):
            supported = ast.literal_eval(node.value)
            break
    if supported is None:
        return None
    version = tuple(int(v) for v in installed.split('.')[:3])
    if version not in supported:
        required = ', '.join('.'.join(map(str, v)) for v in supported)
        return (f'Isaac Sim version mismatch: this Python has {installed}, but the BEHAVIOR checkout '
                f'requires {required}. Align the simulator environment before building or loading Vega.')
    return None
