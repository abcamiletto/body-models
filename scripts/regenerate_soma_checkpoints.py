"""Rebuild native SOMA state_dict checkpoints from configured source assets."""

import argparse
import tempfile
from pathlib import Path

import torch

from body_models.soma import _schema as schema
from body_models.soma.torch import SOMA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-path", type=Path, help="SOMA asset directory; defaults to the configured path.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for lod in schema.SOMA_LODS:
        model = SOMA(model_path=args.model_path, lod=lod)
        destination = args.output_dir / f"soma-{lod}.pt"
        with tempfile.TemporaryDirectory(prefix=f".{destination.stem}-", dir=args.output_dir) as temporary:
            checkpoint = Path(temporary) / destination.name
            torch.save(model.state_dict(), checkpoint)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            restored = SOMA(model_path=args.model_path, lod=lod)
            restored.load_state_dict(saved, strict=True)
            torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
            with torch.no_grad():
                params = model.get_rest_pose()
                expected = model.forward_vertices(**params)
                actual = restored.forward_vertices(**params)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            checkpoint.replace(destination)
        print(destination)


if __name__ == "__main__":
    main()
