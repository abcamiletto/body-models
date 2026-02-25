"""Validate body-models works with minimum dependency versions.

Usage (single package override):
    uv run --with "numpy==2.0.0" python bench/test_min_versions.py

Usage (standalone - tests current env):
    python bench/test_min_versions.py
"""

from __future__ import annotations

import importlib
import sys
import traceback


def tier1_imports() -> bool:
    """Tier 1: All core module imports succeed."""
    modules = [
        "body_models",
        "body_models.anny.numpy",
        "body_models.mhr.numpy",
        "body_models.common",
        "body_models.config",
        "body_models.base",
        "array_api_compat",
        "jaxtyping",
        "nanomanifold",
        "numpy",
        "scipy",
        "platformdirs",
    ]
    ok = True
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"  [OK] import {mod}")
        except Exception as e:
            print(f"  [FAIL] import {mod}: {e}")
            ok = False
    return ok


def tier2_numpy_forward() -> bool:
    """Tier 2: NumPy backend forward pass (ANNY + MHR, auto-download)."""
    ok = True

    # ANNY
    try:
        from body_models.anny.numpy import ANNY

        model = ANNY()
        params = model.get_rest_pose(batch_size=2)
        verts = model.forward_vertices(**params)
        skel = model.forward_skeleton(**params)
        assert verts.shape == (2, model.num_vertices, 3), f"ANNY verts shape: {verts.shape}"
        assert skel.shape[0] == 2 and skel.shape[2:] == (4, 4), f"ANNY skel shape: {skel.shape}"
        print("  [OK] ANNY numpy forward pass")
    except Exception as e:
        print(f"  [FAIL] ANNY numpy forward: {e}")
        traceback.print_exc()
        ok = False

    # MHR
    try:
        from body_models.mhr.numpy import MHR

        model = MHR()
        params = model.get_rest_pose(batch_size=2)
        verts = model.forward_vertices(**params)
        skel = model.forward_skeleton(**params)
        assert verts.shape == (2, model.num_vertices, 3), f"MHR verts shape: {verts.shape}"
        assert skel.shape[0] == 2 and skel.shape[2:] == (4, 4), f"MHR skel shape: {skel.shape}"
        print("  [OK] MHR numpy forward pass")
    except Exception as e:
        print(f"  [FAIL] MHR numpy forward: {e}")
        traceback.print_exc()
        ok = False

    return ok


def tier3_torch_forward() -> bool:
    """Tier 3: PyTorch backend forward pass."""
    try:
        import torch
    except ImportError:
        print("  [SKIP] torch not available")
        return True

    ok = True

    try:
        from body_models.anny.torch import ANNY

        model = ANNY().eval()
        params = model.get_rest_pose(batch_size=2)
        with torch.no_grad():
            verts = model.forward_vertices(**params)
            skel = model.forward_skeleton(**params)
        assert verts.shape == (2, model.num_vertices, 3)
        assert skel.shape[0] == 2 and skel.shape[2:] == (4, 4)
        print("  [OK] ANNY torch forward pass")
    except Exception as e:
        print(f"  [FAIL] ANNY torch forward: {e}")
        traceback.print_exc()
        ok = False

    try:
        from body_models.mhr.torch import MHR

        model = MHR().eval()
        params = model.get_rest_pose(batch_size=2)
        with torch.no_grad():
            verts = model.forward_vertices(**params)
            skel = model.forward_skeleton(**params)
        assert verts.shape == (2, model.num_vertices, 3)
        assert skel.shape[0] == 2 and skel.shape[2:] == (4, 4)
        print("  [OK] MHR torch forward pass")
    except Exception as e:
        print(f"  [FAIL] MHR torch forward: {e}")
        traceback.print_exc()
        ok = False

    return ok


def tier4_jax_forward() -> bool:
    """Tier 4: JAX backend forward pass."""
    try:
        import jax  # noqa: F401
    except ImportError:
        print("  [SKIP] jax not available")
        return True

    ok = True

    try:
        from body_models.anny.jax import ANNY

        model = ANNY()
        params = model.get_rest_pose(batch_size=2)
        verts = model.forward_vertices(**params)
        skel = model.forward_skeleton(**params)
        assert verts.shape == (2, model.num_vertices, 3)
        assert skel.shape[0] == 2 and skel.shape[2:] == (4, 4)
        print("  [OK] ANNY jax forward pass")
    except Exception as e:
        print(f"  [FAIL] ANNY jax forward: {e}")
        traceback.print_exc()
        ok = False

    try:
        from body_models.mhr.jax import MHR

        model = MHR()
        params = model.get_rest_pose(batch_size=2)
        verts = model.forward_vertices(**params)
        skel = model.forward_skeleton(**params)
        assert verts.shape == (2, model.num_vertices, 3)
        assert skel.shape[0] == 2 and skel.shape[2:] == (4, 4)
        print("  [OK] MHR jax forward pass")
    except Exception as e:
        print(f"  [FAIL] MHR jax forward: {e}")
        traceback.print_exc()
        ok = False

    return ok


def main():
    print(f"Python {sys.version}")
    print()

    # Print key package versions
    for pkg in ["numpy", "scipy", "array_api_compat", "nanomanifold", "jaxtyping", "platformdirs"]:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  {pkg}: {ver}")
        except ImportError:
            print(f"  {pkg}: NOT INSTALLED")
    for pkg in ["torch", "jax", "flax"]:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  {pkg}: {ver}")
        except ImportError:
            print(f"  {pkg}: not installed (optional)")
    print()

    all_ok = True

    print("=== Tier 1: Imports ===")
    if not tier1_imports():
        all_ok = False
        print("TIER 1 FAILED - stopping early")
        sys.exit(1)
    print()

    print("=== Tier 2: NumPy forward pass ===")
    if not tier2_numpy_forward():
        all_ok = False
    print()

    print("=== Tier 3: Torch forward pass ===")
    if not tier3_torch_forward():
        all_ok = False
    print()

    print("=== Tier 4: JAX forward pass ===")
    if not tier4_jax_forward():
        all_ok = False
    print()

    if all_ok:
        print("ALL TIERS PASSED")
        sys.exit(0)
    else:
        print("SOME TIERS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
