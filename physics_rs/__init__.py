"""Compiled physics routines, built from `crates/physics-py`.

The functions live in the `_physics_rs` extension module; this package re-exports them so callers
can use `import physics_rs`. Their signatures are documented in `__init__.pyi`.
"""

from ._physics_rs import *  # noqa: F401,F403
