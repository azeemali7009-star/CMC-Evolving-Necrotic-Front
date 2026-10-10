PYTHONPATH=src

.PHONY: data results check clean-generated

data:
	python data/fetch_murphy_data.py

results: data
	PYTHONPATH=src python experiments/run_all.py

check:
	PYTHONPATH=src pytest -q

clean-generated:
	rm -rf results figures

.PHONY: early-audit
early-audit: data
	PYTHONPATH=src python experiments/early_horizon_audit.py
