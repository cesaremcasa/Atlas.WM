"""One-shot migration utility: convert a legacy .pt checkpoint to .safetensors.

Usage:
    python scripts/migrate_pt_to_safetensors.py \
        --input checkpoints/best_model.pt \
        --output checkpoints/best_model.safetensors

The input file is loaded with weights_only=True (safe mode) and re-saved as a
safetensors file with placeholder metadata so it passes load_checkpoint validation.
The original .pt file is NOT deleted; move it to archive/ manually after verifying.

See docs/SECURITY.md — AD-4 forbids .pt files in production paths.
"""

import argparse
import os
import sys

import torch

from atlas_wm.checkpointing.io import make_metadata, save_checkpoint


def migrate(input_path: str, output_path: str) -> None:
    if not input_path.endswith(".pt") and not input_path.endswith(".pth"):
        print(f"ERROR: input must be a .pt or .pth file, got: {input_path!r}", file=sys.stderr)
        sys.exit(1)

    if not output_path.endswith(".safetensors"):
        print(f"ERROR: output must end with .safetensors, got: {output_path!r}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"ERROR: input file not found: {input_path!r}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path!r} with weights_only=True ...")
    raw = torch.load(input_path, map_location="cpu", weights_only=True)

    if not isinstance(raw, dict):
        print(f"ERROR: expected a dict, got {type(raw).__name__}", file=sys.stderr)
        sys.exit(1)

    # Detect nested format: {"encoder": OrderedDict, "dynamics": OrderedDict, ...}
    if "encoder" in raw and "dynamics" in raw and isinstance(raw["encoder"], dict):
        print("Detected nested checkpoint format — flattening encoder.* / dynamics.*")
        state_dict = {
            **{f"encoder.{k}": v for k, v in raw["encoder"].items()},
            **{f"dynamics.{k}": v for k, v in raw["dynamics"].items()},
        }
    else:
        state_dict = raw

    non_tensor_keys = [k for k, v in state_dict.items() if not hasattr(v, "shape")]
    if non_tensor_keys:
        print(f"ERROR: non-tensor values found for keys: {non_tensor_keys}", file=sys.stderr)
        sys.exit(1)

    metadata = make_metadata(
        model_class="ContinuousEncoder+StructuredDynamics",
        config={"migrated_from": input_path},
        env_hash="migrated-from-pt",
    )

    print(f"Saving to {output_path!r} ...")
    save_checkpoint(state_dict, output_path, metadata)
    print(f"Migration complete. Move {input_path!r} to archive/ and update .gitignore.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate .pt checkpoint to .safetensors")
    parser.add_argument("--input", required=True, help="Path to .pt or .pth input file")
    parser.add_argument("--output", required=True, help="Path to .safetensors output file")
    args = parser.parse_args()
    migrate(args.input, args.output)


if __name__ == "__main__":
    main()
