# Dependency and lock policy

Atlas.WM uses `uv.lock` as the canonical cross-platform resolver lock. The
hashed `requirements.lock` is exported from that lock for tools that consume
requirements files (including the CI `pip-audit` gate); it must be regenerated
with:

```bash
uv export --locked --all-extras --format requirements-txt \
  --no-emit-project --output-file requirements.lock
```

`requirements.txt` is retained only as a compatibility shim to that export.
It is not a second set of package pins.

## CPU and GPU compatibility

The project keeps one lock rather than maintaining separate CPU and GPU lock
files. `uv.lock` records the platform and Python markers for the PyTorch CUDA
and Triton dependencies; unsupported platform branches are not installed.
The scientific test suite remains CPU-runnable, while Linux GPU environments
use the standard PyTorch wheel and its resolved CUDA dependencies. Splitting
the lock would create a second dependency graph without changing model code or
results.

## Security floors

- `torch>=2.13.0` removes the audited `PYSEC-2025-194` / `CVE-2025-3000`
  finding reported for the 2.12.1 lock entry.
- `setuptools>=83.0.0` is required for the build backend; the lock currently
  resolves 84.0.0, removing `PYSEC-2026-3447` from the previous 81.0.0 pin.
- Pillow is transitive through the training extra and is resolved to 12.3.0,
  removing the 12.2.0 findings from the stale requirements lock.

No VEX or risk acceptance is recorded for these findings: fixed versions are
available and are used in both tracked locks. `pip-audit` reports package-level
exposure rather than proving call reachability; Atlas.WM does not call
`torch.jit.script` directly, but the vulnerable package is installed, so
suppressing the advisory would not be technically justified while a fixed
version is available.
