# <img src="miso-soup.svg" alt="drawing" width="40"/> MISO: Learning Multiple Initial Solutions to Optimization Problems

This repository contains the code for **[Learning Multiple Initial Solutions to Optimization Problems
](https://openreview.net/forum?id=wsb9GNh1Oi)** by Elad Sharony, Heng Yang, Tong Che, Marco Pavone, Shie Mannor, Peter Karkus.

<p align="center">
  <img src="miso-modules.svg" alt="MISO Modules" />
</p>

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

   If you plan to use the **nuPlan** environment:

   - **Install nuPlan Dataset**

     Follow the [nuPlan installation guide](https://github.com/motional/nuplan-devkit/blob/master/docs/installation.md) to set up the dataset.

   - **Install tuPlan Garage**

     Install [tuPlan Garage](https://github.com/autonomousvision/tuplan_garage) by following their [getting started instructions](https://github.com/autonomousvision/tuplan_garage#getting-started).

   - **Configure Dataset Path**

     Update `NUPLAN_ROOT` in `nuplan/config.py` to point to your nuPlan dataset root directory.


## Usage

This codebase supports all three environments discussed in the paper, cart-pole, reacher, and autonomous driving, using different optimizers: DDP, MPPI, and iLQR.

### Data Preparation

#### 1. Collect Training Data
We first run the warm-start heuristic and then use the problem instances to generate a dataset of (near-)optimal solutions using an oracle proxy:


- **Closed-Loop with Warm-Start**

  ```bash
  python eval.py --env [env_name] --exp closed_loop --method warm_start --eval_set train
  ```

- **Open-Loop with Oracle** 

  ```bash
  python eval.py --env [env_name] --exp open_loop --method oracle --eval_set train
  ```

  Replace `[env_name]` with `cartpole`, `reacher`, or `nuplan`.

#### 2. Generate Dataset

After data collection, generate the dataset required for training:

```bash
python training/dataset.py --env [env_name]
```

The files generated are:
- **Dataset File**: `open_loop_oracle.pth`
- **Scaler File**: `open_loop_oracle_scaler.pkl` (contains standardization statistics)

These files are saved under the `data` directory of the corresponding environment, e.g., `${MISO_ROOT}/envs/[env_name]/data`.


### Model Training

Train the MISO model with the specified parameters:

```bash
python train.py --env [env_name] --num_predictions 16 --miso_method [miso_method] --seed 0
```

`--miso_method` options: 
- `miso-pd`:
Penalize the pairwise distance between all outputs. The overall loss combines this dispersion-promoting term with the regression loss,
```math
\mathcal{L}_{\mathrm{MISO-PD}} = \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}_{\mathrm{reg}}(\mathbf{\hat{x}}_{k}^{\mathrm{init}}, \mathbf{x}^{\star}) +  \alpha_{K} \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}_{\mathrm{PD}, k}(\mathbf{\hat{x}}_{k}^{\mathrm{init}}, \mathbf{x}^{\star}),
```
```math
\mathcal{L}_{\mathrm{PD}, k} = \frac{1}{K-1} \sum_{\substack{k'=1 \\ k' \neq k}}^{K} \Vert \mathbf{\hat{x}}_{k}^{\mathrm{init}} - \mathbf{\hat{x}}_{k'}^{\mathrm{init}} \Vert,
```
where $\alpha_{K}$ is a hyperparameter that balances the trade-off between accuracy and dispersion.

- `miso-wta`:
Select the best-predicted output at training time and only minimize the regression loss for this specific prediction,
```math
 \mathcal{L}_{\mathrm{MISO-WTA}} = \min_{k} \{\mathcal{L}_{\mathrm{reg}}(\mathbf{\hat{x}}_{k}^{\mathrm{init}}, \mathbf{x}^{\star})\}.
```

- `miso-mix`: 
A combination of the previous two approaches to potentially enhance performance, as it provides some measure of dispersion we can tune,
```math
 \mathcal{L}_{\mathrm{MISO-MIX}} = \min_{k} \left\{\mathcal{L}_{\mathrm{reg}}(\mathbf{\hat{x}}_{k}^{\mathrm{init}}, \mathbf{x}^{\star}) +
\alpha_{K} \Phi\left(\mathcal{L}_{\mathrm{PD}, k}(\mathbf{\hat{x}}_{k}^{\mathrm{init}}, \mathbf{x}^{\star}) \right) \right\}, 
```
here, $\Phi$ is an upper-bounded function, such as $\mathrm{min}$ or $\mathrm{tanh}$, designed to limit the contribution of the pairwise distance term.

- `none`:
A simple regression loss without. For $K>1$, the loss of each prediction is summed up.

### Model Evaluation

Evaluate the trained model on the test set:

```bash
python eval.py --env [env_name] --exp closed_loop --method [miso_method] --optimizer_mode [optimizer_mode] --eval_set test
```
- `--optimizer_mode` options:
  - `single`
  - `multiple`
- Each training session runs for **125 epochs**, consistent with the settings in the paper.
- The model reads data from preconfigured paths specified in `training/configs/dataset.yaml`, these paths correspond to where the data is generated during the data preparation steps. **No changes are needed** unless you have customized the data directories.


### Results and Reproducibility
While efforts have been made to ensure reproducibility, results may slightly vary from those reported in the paper, especially in the nuPlan environment with the PDM planner.
**Yet**, the **relative improvements** between different methods and baselines should remain consistent.


## Citation

If you find this project useful in your research, please cite:

```bibtex
TODO
```
