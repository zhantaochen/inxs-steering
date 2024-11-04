# Code repository for Implicit neural representations for experimental steering of advanced experiments

This is the code repository for "Implicit neural representations for experimental steering of advanced experiments". Please direct questions to Zhantao Chen (zhantao@stanford.edu).

The names of notebooks are self-explanatory, with data necessary for reproducing the reported results shared at TBD.

If you are interested in running benchmarking, you could
```
conda activate your_env

python steering_real_exp_serial.py --config-name config_poisson
```

Key required packages are listed below:
```
hydra-core==1.3.2
lightning==2.3.3
matplotlib==3.9.2
numpy==1.26.4
scikit-learn==1.5.1
scipy==1.14.0
seaborn==0.13.2
torch==2.3.1
```