
## Installation

### Operating system information
Ubuntu 18.04. A large VM is preferred, e.g., reproducing Figure 9 CC takes
about 20 minutes on a VM with 96 vCPUs or 1 hour on a VM with 32 vCPUs. We
assume a VM with 32 vCPUs, 64G memory and 32G SSD storage is used for the
instructions below.

### Python Version
The repository is only tested under python3.6

### Install python dependency
Time usage: 2~3min on a VM with 32 vCPUs
```bash
# activate virtual env
bash install_python_dependency.sh
```

### ABR
Time usage: 1~2 min on a VM with 32 vCPUs.

### CC
To get a fast and slightly different but with the correct trend,
the following command run models with 1 seed on 50 synthetic traces, 

Time usage: 2 min on a VM with 32 vCPUs.
```bash
cd FenNR # cd into the project root
python sr/simulator/evaluate_synthetic_traces.py \
  --save-dir results/cc/evaluate_synthetic_dataset \
  --dataset-dir datas/cc/synthetic_dataset \
  --fast
python sr/plot_scripts/plot_syn_dataset.py
```

To get a complete results,  the following commands run models with 5 different
seeds on ~500 synthetic traces

Time usage: 60 min on a VM with 32 vCPUs.
```bash
cd FenNR # cd into the project root
python sr/simulator/evaluate_synthetic_traces.py \
  --save-dir results/cc/evaluate_synthetic_dataset \
  --dataset-dir datas/cc/synthetic_dataset
python sr/plot_scripts/plot_syn_dataset.py
```


## FAQ
1. CUDA driver error

    If the following cuda driver error message shows up, please ignore for now.
    The final results are not affected by the error message.

