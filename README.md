# DeepSLEEP
Strong Lensing parameter inference with SBI

Current ground-based cosmological surveys, such as the Dark Energy Survey (DES), are predicted to discover thousands of galaxy-scale strong lenses, while future surveys, such as the Vera Rubin Observatory Legacy Survey of Space and Time (LSST) will increase that number by 1-2 orders of magnitude.
The large number of strong lenses discoverable in future surveys will make strong lensing a highly competitive and complementary cosmic probe.

To leverage the increased statistical power of the lenses that will be discovered through upcoming surveys, automated lens analysis techniques are necessary. We present two Simulation-Based Inference (SBI) approaches for lens parameter estimation of galaxy-galaxy lenses. We demonstrate the successful application of Neural Posterior Estimation (NPE) to automate the inference of a 12-parameter lens mass model for DES-like ground-based imaging data. We compare our NPE constraints to a Bayesian Neural Network (BNN) and find that it outperforms the BNN, producing posterior distributions that are for the most part both more accurate and more precise; in particular, several source-light model parameters are systematically biased in the BNN implementation.

Here we provide implementation for both NPE and BNN approaches presented in our paper.
