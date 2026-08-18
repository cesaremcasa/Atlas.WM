.PHONY: install lint type test security ci-local sbom release-staging

install:
	uv sync --extra dev

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

type:
	mypy src/atlas_wm

test:
	pytest -q --cov=atlas_wm --cov-report=xml

security:
	bandit -r src/ -ll
	pip-audit -r requirements.lock --progress-spinner off
	@! grep -rE "torch\.load|pickle\.load|torch\.save" src/atlas_wm/ \
		$$(find scripts/ -name "*.py" ! -name "migrate_pt_to_safetensors.py") \
		|| (echo "ERROR: unsafe pickle/torch.save in prod code" && exit 1)

ci-local: lint type test security
	@echo "All CI checks passed locally."

sbom:
	python scripts/generate_sbom.py --project pyproject.toml --lock requirements.lock --output sbom.json
	@echo "SBOM written to sbom.json"

release-staging:
	@stage="$${RELEASE_OUTPUT_DIR:-$$(mktemp -d "$${TMPDIR:-/tmp}/atlas-release.XXXXXX")}"; \
	uv run --locked --python 3.11.15 --extra dev python scripts/build_release.py --output-dir "$$stage"; \
	echo "Release staging written to $$stage"
