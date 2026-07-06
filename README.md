## How to Run
### 1. Setup repository
Clone the repository and move into the project directory:

```bash
git clone <repo-url>
cd TTEL
```
### 2. Define experiment configurations
Edit `experiments.csv` to specify your experiment settings.

The file includes a few example configurations (one per dataset).  
To add new experiments, simply duplicate an existing row and modify the hyperparameters as needed.

### 3. Run experiments
Execute all configured experiments using:

```bash
./run.sh
```

### 4. Analyze results

Open and run `results.ipynb` to aggregate and visualize results.

The notebook builds a unified dataframe over all experiments, enabling systematic analysis of the saved metrics across runs.

