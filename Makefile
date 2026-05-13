.PHONY: install lint type test security ci-local sbom

install:
	pip install -e ".[dev]"

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
	@! grep -rE "torch\.load|pickle\.load|torch\.save" src/atlas_wm/ scripts/ \
		|| (echo "ERROR: unsafe pickle/torch.save in prod code" && exit 1)

ci-local: lint type test security
	@echo "All CI checks passed locally."

sbom:
	cyclonedx-py environment -o sbom.json
	@echo "SBOM written to sbom.json"
