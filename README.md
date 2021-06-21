# MUSTEM-Model-using-NBI
This project utilized the multi-stage exponential Markov (MUSTEM) model for deterioration modeling of highway bridge components (decks,supserstructures, and substructures) using the National Bridge Inventory database (1992-2019).

The original MUSTEM model technique was developped by

Tsuda, Y., Kaito, K., Aoki, K., & Kobayashi, K. (2006). Estimating Markovian transition probabilities for bridge deterioration forecasting. Structural Engineering/Earthquake Engineering, JSCE, 23(2), 241s-256s. doi:https://doi.org/10.2208/jsceseee.23.241s

The model was compiled using MATLAB. To run the code:

1. Add the folder of "MUSTEM_functions" to the Path. This folder includes necessary function files to develop the MUSTEM model. For example, the "MUSTEM_Loss_wGrad" calculates the optimization loss and corresponding gradient with given training samples. The gradient functions were generated using "MUSTEM_Loss_wGrad_funGenrator_newton.mat".

2. Go to a component file: Deck, Super, or Sub.

3. Run "A_MUSTEM_quasi_Newton.m" to calibrate the model coefficients. Pre-calculated model coefficients were stored in files from "Beta_quasiNewton_fold_1.mat" to "Beta_quasiNewton_fold_10.mat".

4. Run "B_MUSTEM_Evaluation.m" to evlauate the calibrated model on an independent testing dataset. Pre-calculated testing results were stored in files "error_MUSTEM_deck.mat", "error_MUSTEM_super.mat", and "error_MUSTEM_sub.mat". The optimization utilized the quasi-Newton method. 


The folder of "Samples" under each component file contains the data used for the model development. 

The files "Feature_Matrix_fold_1.mat" to "Feature_Matrix_fold_10.mat" contain the original data for the 10-fold cross validation. The pre-shuffled train/test splits were also included. 

The files "Samples_Labels_fold_1.mat" to "Samples_Labels_fold_10.mat" contain the training data for model development. The data was discretized from the original data in the corresponding fold. Samples with a current condition rating of 3 was not included since the transition probability was known and assumed to be 1.
