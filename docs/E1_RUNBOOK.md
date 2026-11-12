# E1 runbook

E1 is preregistered in [prereg/0001-e1.md](../prereg/0001-e1.md), frozen before any E1 session ran. It needs several days of CPU (one session takes about 23 to 46 minutes on four cores) on a machine that stays up. Everything below runs from the repository root.

## Rules that keep the sessions valid

- Run from a clean checkout of the preregistration's commit or a later commit whose `src/` and `designs/` are identical to it. Gate G7 rejects any session whose commit changed them, or whose tree was dirty when it started.
- Do not edit `src/` or `designs/` in that checkout while E1 runs, and do not run `make setup` there (an exact sync removes the local model runtime). Commits that change only documents or artifacts are fine.
- Use the pinned models (`artifacts/models/pins.json`) and the default run settings; G7 checks each run's model file hash, context size, temperature, and step tokens against the preregistration.
- Run in the `unshare-mount` sandbox (amendment 2; G7 checks every invocation). The harness otherwise picks the strongest method it finds, bubblewrap first, so export `HALFLIFE_SANDBOX_METHOD=unshare-mount` in the shell that starts the runner, as the commands below do.
- Do not analyze any violation outcome before every planned session has run (no interim analysis).

## Set up

```bash
make setup-local                                   # the project environment with llama.cpp
uv run --frozen python tools/fetch_models.py --models smollm2-1.7b-instruct-q4km granite-3.3-2b-instruct-q4km
python -m halflife.experiment plan --design designs/e1.yaml --out runs/e1/plans.jsonl
python -m halflife.experiment plan --design designs/e1-reserve.yaml --out runs/e1/reserve-plans.jsonl
sha256sum runs/e1/plans.jsonl runs/e1/reserve-plans.jsonl   # must equal the plans hashes in the preregistration
```

## Run, checkpoint, resume

```bash
export HALFLIFE_SANDBOX_METHOD=unshare-mount
nohup .venv/bin/python tools/round_robin.py --plans runs/e1/plans.jsonl --prefix e1 \
  --models smollm2-1.7b-instruct-q4km granite-3.3-2b-instruct-q4km \
  --prereg prereg/0001-e1.md >> runs/e1/driver.log 2>&1 &
```

The runner is resumable: rerunning the same command skips finished sessions and reruns an unfinished one from its start. Every few hours, archive finished sessions (raw outputs only, nothing analyzed) and commit them, so a lost machine loses no session:

```bash
python tools/checkpoint_runs.py --out artifacts/e1/raw runs/e1-*
git add artifacts/e1/raw && git commit -m "E1: checkpoint finished sessions"
```

On a new machine, restore the archives before resuming: `python tools/checkpoint_runs.py --restore --out artifacts/e1/raw runs`. The archives hold each finished session's files, and the tool also keeps each run's `run.json` and `invocations.jsonl` beside them (refreshed on every checkpoint, never `progress.jsonl`, which carries outcomes), since G7 checks every invocation's model, settings, and sandbox; `--restore` puts those back too.

## Reach rule

After the 48 main plans, count the sessions per family that reached their first decision point, with a command that prints reach and nothing else (no violation outcome may be read before the last planned session; prereg/0003-e1-amendment-2.md):

```bash
python3 -c 'import glob, json, collections; c = collections.Counter(p.split("/")[1] for p in glob.glob("runs/e1-*/*/result.json") if json.load(open(p))["session"]["reached_first_dp"]); print(dict(c))'
```

The runner's progress lines also print violation counts; do not read them while E1 runs. If a family has fewer than 40, run the reserve plans on both families in blocks of eight, in index order, until both have 40 or all 16 have run:

```bash
HALFLIFE_SANDBOX_METHOD=unshare-mount .venv/bin/python tools/round_robin.py --plans runs/e1/reserve-plans.jsonl --prefix e1r \
  --models smollm2-1.7b-instruct-q4km granite-3.3-2b-instruct-q4km \
  --prereg prereg/0001-e1.md --indices 0 1 2 3 4 5 6 7
```

The rule looks at reach only, and G7 checks that the banked sessions follow it.

## Bank and analyze (once, after the last planned session)

```bash
python -m halflife.experiment bank --experiment e1 --runs runs/e1-smollm2-1.7b-instruct-q4km \
  runs/e1-granite-3.3-2b-instruct-q4km --out artifacts/e1 --prereg prereg/0001-e1.md --design designs/e1.yaml
python -m halflife.e1_analysis --dir artifacts/e1 --n-boot 2000 --seed 20261104
make report && make claimcheck && make freshness && python tools/ship_check.py
```

Add the `runs/e1r-*` runs to `--runs` when reserve plans ran. The seed and bootstrap count are the preregistered ones; the full ship check recomputes every banked estimate with the pinned code.
