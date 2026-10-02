.PHONY: setup smoke test audit aggregate clean-smoke

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -U pip
	.venv/bin/python -m pip install -e '.[test]'

smoke:
	PYTHONPATH=src python3 -m oap_supcon.cli make-smoke
	OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m oap_supcon.cli run --dataset smoke --method oap_supcon --seed 11 --epochs 2 --device cpu --corruption-realizations 1

test:
	PYTHONPATH=src python3 -m pytest -q

audit:
	PYTHONPATH=src python3 -m oap_supcon.cli audit --dataset smoke

aggregate:
	PYTHONPATH=src python3 -m oap_supcon.cli aggregate

clean-smoke:
	find data/smoke/processed -type f -name '*.npz' -delete
	find results_v2 -mindepth 1 -maxdepth 1 -type d -name 'smoke_*' -exec rm -r {} +
