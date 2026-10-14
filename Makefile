PYTHON ?= python
RESULTS ?= results
export TSSEG_ANALYSIS_DB ?= $(CURDIR)/$(RESULTS)/results.db

.PHONY: data results-db paper test lint

## Every dataset, into data/ (Hugging Face Hub, or the source for USC-HAD and UTSA).
data:
	$(PYTHON) scripts/download_data.py

## Rebuild the SQLite database read by the analysis from the results tables.
results-db: $(RESULTS)/results.db

$(RESULTS)/results.db: $(RESULTS)/runs.parquet $(RESULTS)/configs.parquet
	$(PYTHON) scripts/build_results_db.py --results $(RESULTS)

## Every figure and table of the paper, in figures/paper/.
paper: $(RESULTS)/results.db
	cd benchmark && $(PYTHON) -m analysis.paper

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check . && ruff format --check .
