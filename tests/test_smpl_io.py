import pickle
import sys
import types

import numpy as np
from scipy.sparse import csc_matrix

from body_models.smpl.io import load_model_data


def test_load_model_data_handles_chumpy_pickle(tmp_path, monkeypatch) -> None:
    array = np.arange(6, dtype=np.float32).reshape(2, 3)

    chumpy_pkg = types.ModuleType("chumpy")
    chumpy_mod = types.ModuleType("chumpy.ch")

    class Ch:
        __module__ = "chumpy.ch"

        def __getstate__(self):
            return {"x": array}

        def __setstate__(self, state):
            pass

    Ch.__qualname__ = "Ch"
    chumpy_mod.Ch = Ch
    chumpy_pkg.ch = chumpy_mod

    monkeypatch.setitem(sys.modules, "chumpy", chumpy_pkg)
    monkeypatch.setitem(sys.modules, "chumpy.ch", chumpy_mod)

    model_path = tmp_path / "SMPL_NEUTRAL.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"v_template": Ch(), "J_regressor": csc_matrix(np.eye(2, dtype=np.float32))}, f, protocol=2)

    model_data = load_model_data(model_path)

    np.testing.assert_array_equal(model_data["v_template"], array)
    np.testing.assert_array_equal(model_data["J_regressor"], np.eye(2, dtype=np.float32))
