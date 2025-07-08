# LineTW
Official project repository of 'Learning Link-Centric Temporal Representations via Graph Structure Transformation for Continuous-Time Dynamic Link Prediction'

Most of the used original dynamic graph datasets come from (https://openreview.net/forum?id=1GVpwr2Tfdg), 
which can be downloaded [here](https://zenodo.org/record/7213796#.Y1cO6y8r30o). 
Please download them and put them in ```DG_data``` folder. 
We can run ```preprocess_data/preprocess_data.py``` for pre-processing the datasets.
For example, to preprocess the *Wikipedia* dataset, we can run the following commands:
```{bash}
cd preprocess_data/
python preprocess_data.py  --dataset_name wikipedia
```
We can also run the following commands to preprocess all the original datasets at once:
```{bash}
cd preprocess_data/
python preprocess_all_data.py

## Training and Testing
See the './src/run.sh' file

## Environments

[PyTorch 1.8.1](https://pytorch.org/),
[numpy](https://github.com/numpy/numpy),
[pandas](https://github.com/pandas-dev/pandas),
[tqdm](https://github.com/tqdm/tqdm), and 
[tabulate](https://github.com/astanin/python-tabulate)