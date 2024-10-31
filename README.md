# <img src="miso-soup.svg" alt="drawing" width="40"/> MISO: Learning Multiple Initial Solutions to Optimization Problems

---

This repository contains the code for **[Learning Multiple Initial Solutions to Optimization Problems
](https://openreview.net/forum?id=wsb9GNh1Oi)** by Elad Sharony, Heng Yang, Tong Che, Marco Pavone, Shie Mannor, Peter Karkus.

<div style="text-align: center;">
  <img src="miso-modules.svg" alt="MISO Modules" />
</div>

The main idea of &nbsp;<img src="miso-soup.svg" alt="drawing" width="15"/> **MISO** is to train a single neural network to predict *multiple* initial solutions to
an optimization problem, such that the initial solutions cover promising regions of the optimization
landscape, eventually allowing a local optimizer to find a solution close to the global optimum.

---

## Installation

To set up the project environment, follow these steps:

1. **Update Paths**

   Run the `setup.py` script to configure necessary paths:

   ```bash
   python setup.py
   ```

2. **Install Dependencies**

   Create and activate the Conda environment using `environment.yml`:

   ```bash
   conda env create -f environment.yml
   conda activate miso
   ```

3. **Set Up nuPlan**

   - **Install nuPlan Dataset**

     Follow the [nuPlan installation guide](https://github.com/motional/nuplan-devkit/blob/master/docs/installation.md) to set up the dataset.

   - **Install tuPlan Garage**

     Install [tuPlan Garage](https://github.com/autonomousvision/tuplan_garage) by following their [getting started instructions](https://github.com/autonomousvision/tuplan_garage#getting-started).

   - **Configure Dataset Path**

     Update `NUPLAN_ROOT` in `nuplan/config.py` to point to your nuPlan dataset root directory.

---

## Usage

### Data preparation

1. **Closed-Loop with Warm-Start**

   Collect training data using closed-loop evaluation:

   ```bash
   python eval.py --env nuplan --exp closed_loop --method warm_start --eval_set train
   ```

2. **Open-Loop with Oracle**

   Collect data using open-loop evaluation with oracle guidance:

   ```bash
   python eval.py --env nuplan --exp open_loop --method oracle --eval_set train
   ```

3. Generate the dataset required for training:
   ```bash
   python training/dataset.py --env nuplan
   ```

### Model Training

Train the MISO model with the specified parameters:

```bash
python train.py --env nuplan --num_predictions 16 --miso_method miso-wta --seed 0
```

### Model Evaluation

Evaluate the trained model on the test set:

```bash
python eval.py --env nuplan --exp closed_loop --method miso-wta --eval_set test
```

---

## Citation

If you find this project useful in your research, please cite:

```bibtex
TODO
```
