import importlib.util
import os

# Import the compiled extension module
_module_name = None
for fname in os.listdir(os.path.dirname(__file__)):
    if fname.startswith('_core') and fname.endswith('.so'):
        _module_name = fname
        break

if _module_name:
    spec = importlib.util.spec_from_file_location(
        "_core",
        os.path.join(os.path.dirname(__file__), _module_name)
    )
    _core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_core)
    
    # Export all from _core
    for name in dir(_core):
        if not name.startswith('__'):
            globals()[name] = getattr(_core, name)
    
    __doc__ = _core.__doc__ if hasattr(_core, "__doc__") else None
    if hasattr(_core, "__all__"):
        __all__ = _core.__all__
else:
    raise ImportError("Could not find compiled _core extension")