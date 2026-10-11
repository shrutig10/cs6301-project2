# CS 6301 Project 2: CNNs and Residual Learning

CIFAR-10 image classification using custom PyTorch models. This project
implements a baseline CNN, studies data augmentation, and compares
matched plain and residual deep CNNs.

## Environment

Experiments were run using:

- macOS on an Apple M5 Pro with 24 GB memory
- Python 3.13.16
- PyTorch with the MPS GPU backend
- Python package versions recorded in `requirements.txt`

Training and evaluation automatically use MPS when available and
otherwise use the CPU. CPU execution may take longer.

## Project structure

```text
src/              Python source code
data/             CIFAR-10 files, fixed split, normalization statistics
checkpoints/      Best-validation checkpoints (excluded from Git)
results/          Experiment logs, summaries, and learning curves
requirements.txt  Recorded dependency versions
```

## Setup

Open a terminal in the repository's root directory. All commands below
must be run from this directory.

If Python 3.13 is not installed, install it using Homebrew:

```bash
brew install python@3.13
```

Create and activate the project environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the Python version:

```bash
python --version
```

The original environment used Python 3.13.16. Homebrew may install a
newer patch release of Python 3.13.

In VS Code, open **Python: Select Interpreter** from the Command Palette
and select the interpreter inside `.venv`.

When starting a new terminal session, activate the environment again:

```bash
source .venv/bin/activate
```

## Execution instructions

Run the following steps in order. Execute each training command
separately and wait for it to finish before starting the next.

### 1. Prepare the dataset

```bash
python src/prepare_data.py
```

This downloads CIFAR-10 and creates or reloads the fixed split:

- 45,000 training examples
- 5,000 validation examples
- Split seed 42

It saves `data/split_seed42.json` and computes training-only channel
statistics in `data/normalization.json`. Every experiment reuses these
indices and normalization statistics.

### 2. Run sanity checks

```bash
python src/baseline.py
python src/deep_models.py
python src/sanity_checks.py
python src/tiny_overfit.py
```

These verify model shapes, uniform-prediction loss, finite gradients,
weight updates, and at least 95% training accuracy on 64 clean examples.

Tiny-set results are saved in `results/tiny_overfit.json`.

### 3. Run Part B: baseline augmentation comparison

Without augmentation:

```bash
python src/train_baseline.py --seed 0
python src/train_baseline.py --seed 1
python src/train_baseline.py --seed 2
```

With random cropping and horizontal flipping:

```bash
python src/train_baseline.py --seed 0 --augment
python src/train_baseline.py --seed 1 --augment
python src/train_baseline.py --seed 2 --augment
```

Generate summary statistics and figures:

```bash
python src/plot_part_b.py
```

Figure outputs:

```text
results/partB_learning_curves.png
results/partB_learning_curves.pdf
```

### 4. Run Part C: plain versus residual comparison

Plain model:

```bash
python src/train_deep.py --model plain --seed 0
python src/train_deep.py --model plain --seed 1
python src/train_deep.py --model plain --seed 2
```

Residual model:

```bash
python src/train_deep.py --model residual --seed 0
python src/train_deep.py --model residual --seed 1
python src/train_deep.py --model residual --seed 2
```

Both models use the same training augmentation.

Generate summary statistics and figures:

```bash
python src/plot_part_c.py
```

Figure outputs:

```text
results/partC_learning_curves.png
results/partC_learning_curves.pdf
```

### 5. Evaluate the selected configuration

Complete all training and validation comparisons before this step.
The selected configuration is the residual CNN, based on mean
best-validation accuracy across seeds 0, 1, and 2.

The following checkpoints must exist:

```text
checkpoints/partC_residual_seed0.pt
checkpoints/partC_residual_seed1.pt
checkpoints/partC_residual_seed2.pt
```

Run:

```bash
python src/evaluate_test.py
```

This evaluates all three checkpoints on the official 10,000-image test
set without augmentation. Results are saved to:

```text
results/final_test_results.json
```

Final test accuracy: **83.53% ± 0.17 percentage points**
(mean ± sample standard deviation across the three seeds).

Test results are not used to change model or training settings.

## Training settings and saved outputs

All main comparison runs use:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Batch size | 64 |
| Epoch budget | 20 |
| Training seeds | 0, 1, 2 |
| Dataset split seed | 42 |

The scripts record training loss, validation loss, training accuracy,
and validation accuracy per epoch. Losses are averaged with weighting
by the number of examples in each batch.

Each run saves its log in `results/` and its best-validation checkpoint
in `checkpoints/`. Higher validation accuracy takes priority; lower
validation loss breaks ties.

Training scripts refuse to overwrite existing logs or checkpoints.
To intentionally repeat a run, first archive its corresponding files.
Interrupted-training resumption is not implemented.

Final test evaluation also refuses to overwrite an existing results
file. Read the saved JSON to inspect previously computed results.

Downloaded dataset files, the virtual environment, and checkpoints are
excluded from Git. A fresh clone requires data preparation and training
before test evaluation. Learning curves can be regenerated directly
from existing experiment logs.