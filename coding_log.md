# Coding log — commands

Everything runs from the project root (`self/`), scripts as modules.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# after adding a package
pip freeze > requirements.txt
```

## API client

```powershell
python -m scripts.check_client                          # default tier, fallback on
python -m scripts.check_client --no-fallback            # fail instead of falling back
python -m scripts.check_client --tier fast              # strong | medium | fast
python -m scripts.check_client --tier medium --no-fallback

python -m scripts.check_mock                            # offline canned client

Get-Content experiments\results\api_log.jsonl           # per-call log
```

## Data (Phase 2)

```powershell
python -m scripts.check_loaders --dataset hatexplain    # hatexplain | mhs | implicit_hate | gahd | detox
python -m scripts.mapping_coverage                      # dimension x dataset matrix
python -m scripts.plots.plot_mhs                        # severity histogram + cut-points
python -m scripts.make_splits                           # rebuild frozen splits (seed 42)
```

## Knowledge base (Phase 3)

```powershell
python -m scripts.check_taxonomy                        # every label has a definition
python -m scripts.build_kb_records                      # taxonomy + guidelines + examples -> kb/records.jsonl
python -m scripts.check_kb_schema                       # validate records.jsonl
python -m scripts.build_kb                              # records.jsonl -> ChromaDB (idempotent)
```

## Retrieval (Phase 4)

```powershell
python -m scripts.debug_retrieval                       # all 6 standard probes
python -m scripts.debug_retrieval --strategy bm25       # dense | bm25 | hybrid
python -m scripts.debug_retrieval "some text" --lang de # single query
```

## Tests

```powershell
pytest -v                                               # full suite
pytest tests/test_retrieval_probes.py -v                # one file
```



## Notes

- `pytest.ini` restricts collection to `tests/`, so `scripts/check_*.py` are ignored.
- Rebuild the KB (`build_kb_records` then `build_kb`) after ANY edit to
  `taxonomy.yaml` or `guidelines.yaml`.