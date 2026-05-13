"""Sign all .safetensors checkpoints in a directory and write manifest.sig.

Usage:
    ATLAS_SIGNING_KEY=<hex_key> python scripts/sign_checkpoint.py --dir checkpoints/

The signing key must be a 64-character hex string (32 bytes of entropy).
Generate a key once with:
    python -c "import secrets; print(secrets.token_hex(32))"
Store it in your environment (CI secrets, .env, etc.). Never commit it.

After signing, commit checkpoints/manifest.sig. The key stays secret.
Never edit manifest.sig by hand — always regenerate via this script.
"""

import argparse
import os
import sys

from atlas_wm.checkpointing.signing import build_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign Atlas.WM checkpoints with HMAC-SHA256")
    parser.add_argument(
        "--dir",
        default="checkpoints",
        help="Directory containing .safetensors files (default: checkpoints/)",
    )
    parser.add_argument(
        "--key-env",
        default="ATLAS_SIGNING_KEY",
        help="Environment variable holding the hex-encoded 32-byte signing key",
    )
    args = parser.parse_args()

    hex_key = os.environ.get(args.key_env, "")
    if not hex_key:
        print(
            f"ERROR: {args.key_env} is not set. "
            'Generate a key with: python -c "import secrets; print(secrets.token_hex(32))"',
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        key = bytes.fromhex(hex_key)
    except ValueError:
        print(f"ERROR: {args.key_env} is not valid hex.", file=sys.stderr)
        sys.exit(1)

    if len(key) < 16:
        print("ERROR: signing key must be at least 16 bytes (32 hex chars).", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.dir):
        print(f"ERROR: directory not found: {args.dir!r}", file=sys.stderr)
        sys.exit(1)

    safetensors_files = [f for f in os.listdir(args.dir) if f.endswith(".safetensors")]
    if not safetensors_files:
        print(f"No .safetensors files found in {args.dir!r}.", file=sys.stderr)
        sys.exit(1)

    print(f"Signing {len(safetensors_files)} checkpoint(s) in {args.dir!r} ...")
    manifest = build_manifest(args.dir, key)

    manifest_path = os.path.join(args.dir, "manifest.sig")
    write_manifest(manifest_path, manifest)

    print(f"Written: {manifest_path}")
    for filename in manifest["files"]:
        print(f"  {filename}: {manifest['files'][filename][:16]}...")
    print("Done. Commit manifest.sig. Keep the signing key secret.")


if __name__ == "__main__":
    main()
