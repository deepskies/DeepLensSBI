# Deep inference of simulated strong lenses in ground-based surveys
Current ground-based cosmological surveys, such as the Dark Energy Survey (DES), are predicted to discover thousands of galaxy-scale strong lenses, while future surveys, such as the Vera Rubin Observatory Legacy Survey of Space and Time (LSST) will increase that number by 1-2 orders of magnitude.
The large number of strong lenses discoverable in future surveys will make strong lensing a highly competitive and complementary cosmic probe.

To leverage the increased statistical power of the lenses that will be discovered through upcoming surveys, automated lens analysis techniques are necessary. We present two Simulation-Based Inference (SBI) approaches for lens parameter estimation of galaxy-galaxy lenses. We demonstrate the successful application of Neural Posterior Estimation (NPE) to automate the inference of a 12-parameter lens mass model for DES-like ground-based imaging data. We compare our NPE constraints to a Bayesian Neural Network (BNN) and find that it outperforms the BNN, producing posterior distributions that are for the most part both more accurate and more precise; in particular, several source-light model parameters are systematically biased in the BNN implementation.

Here we provide implementation for both NPE and BNN approaches presented in our paper.

# Installation
1. Set up an environment. This can be done using

```
conda create --name deeplensSBI python=3.9
conda activate deeplensSBI
```


2. Install the dependencies needed for training the model provided in requirements.txt. 
```
pip install --user -r "requirements.txt"
```

3. To train a NPE model, run
```
python src/sbi_runner.py --num_params --hidden_features --num_transforms --out_features --seed
```
The arguments of the model are:

| Argument      | Description |
| ----------- | ----------- |
| num_params      | Which model you are training. Options are '1', '5' and '12'. |
| hidden_features   | Sets the number of hidden units used in the MAF model |
| num_transforms      | Sets the number of flow transformations used in the MAF model  |
| out_features   | Sets the number of output features the embedding network outputs |

The script will output a pickle file that contains the trained neural posterior estimator. 

4. To train a BNN model, see the script in src/12_param_BNN.py as an example of how to train a 12-parameter BNN. This script can be adapted for training a 1- or 5-parameter model.

Examples of how to use the trained neural posterior estimator can be found in the analysis notebooks in the "Analysis" folder. These notebooks document how to produce the plots that are published in the paper. The files required to run the notebooks can be found in the Zenodo repository for this project: https://zenodo.org/records/13961234.
