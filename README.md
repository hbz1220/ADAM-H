# ADAM-H

Reproducibility code and verified numerical results for the manuscript

> **ADAM-H: Decoupled Adaptive High-Index Saddle Dynamics with Stochastic Derivative-Free Extensions**

ADAM-H decouples the two roles of the gradient in adaptive high-index saddle dynamics: the Householder-reflected gradient drives the first moment, while the original gradient drives the second moment. The repository contains the scripts and raw outputs for all experiments in Section 7 of the manuscript.

## Repository structure

```text
.
├── code/
│   ├── 7.1.py
│   ├── 7.2.py
│   ├── 7.3.py
│   ├── 7.4.py
│   ├── 7.5.py
│   ├── 7.6.py
│   ├── 7.7.py
│   ├── optimizers.py
│   ├── derivative_free_optimizers.py
│   ├── exact_adamh_baselines.py
│   └── experiment_utils.py
└── result/
    ├── 7.1/
    ├── 7.2/
    ├── 7.3/
    ├── 7.4/
    ├── 7.5/
    ├── 7.6/
    └── 7.7/
```

The numbered scripts correspond directly to Sections 7.1--7.7. Shared optimizer implementations and validation utilities remain in `code/` and are imported by the numbered scripts.

## Requirements

- Python 3.11 or later
- NumPy
- PyTorch
- Matplotlib (Section 7.2 only)

After downloading or cloning the repository, create an isolated environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All reported experiments use CPU execution by default. No external datasets are required.

## Reproducing the numerical experiments

Run commands from the repository root. Each script writes its outputs to `result/7.x/` by default.

| Manuscript section | Experiment | Command |
|---|---|---|
| 7.1 | Exact-gradient stress tests and controls | `python code/7.1.py` |
| 7.2 | Stochastic and second-moment diagnostics | `python code/7.2.py` |
| 7.3 | Derivative-free condition-number crossover | `python code/7.3.py` |
| 7.4 | Additive stochastic-gradient noise | `python code/7.4.py` |
| 7.5 | Rotated quartic benchmarks and auxiliary step-size sweep | `python code/7.5.py --fresh` |
| 7.6 | Exact-gradient Allen--Cahn benchmark | `python code/7.6.py` |
| 7.7 | Randomly rotated anisotropic quadratic | `python code/7.7.py` |

### Optional modes

Section 7.1 can run its components separately:

```bash
python code/7.1.py --mode stress
python code/7.1.py --mode direction-control
python code/7.1.py --mode eig-regime-map
```

Its default `--mode all` generates the three reported CSV files. The optional `--full-grid` flag runs the larger exact-eigensolver regime map, which is not required for the reported tables.

Section 7.2 also provides separate modes:

```bash
python code/7.2.py --mode fixed-state
python code/7.2.py --mode second-moment
```

Its default `--mode all` runs both diagnostics.

Section 7.5 provides separate modes for the higher-index panel, condition-number panel, and auxiliary step-size sweep:

```bash
python code/7.5.py --mode higher-index --fresh
python code/7.5.py --mode condition-number --fresh
python code/7.5.py --mode stepsize --fresh
```

Without `--fresh`, Section 7.5 may reuse compatible rows already present in the committed result files. Use `--fresh` for an independent rerun.

## Reproducibility notes

- Random seeds, sampling order, stopping rules, certification criteria, and CSV schemas are fixed in the scripts.
- In derivative-free experiments, true gradients and Hessians are used only for stopping, final-index certification, and the disclosed pilot rankings; they are not used in the optimizer updates.
- The committed files under `result/` are the verified raw outputs used for the manuscript tables and quantitative statements.
- Section 7.5 tests DF-HiSD step sizes in `{0.001, 0.005, 0.01, 0.02, 0.05, 0.1}` on five pilot starts disjoint from the reported starts. Candidates share initial states and estimator streams and are selected lexicographically by decreasing certified-success count, then increasing successful-run median, summed final true-gradient norm, and step size.
- Some full experiments, particularly Sections 7.3, 7.5, and 7.7, require many optimizer runs and may take substantial time.
- Running a script may overwrite files with the same names in its default result directory. Preserve a clean copy of the committed results when performing comparisons.

## Results

The `result/` directory contains the formal CSV outputs, pilot-sweep records used for step-size selection, auxiliary controls discussed in the manuscript, and the Section 7.2 diagnostic figure.

## License

This repository is released under the MIT License; see `LICENSE`.