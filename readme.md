# 1. Introduction

This practice aims to construct machine learning models (QSAR model) for virtual screening and drug discovery. **A regression and a classification model were constructed.**

The ANN model coupled with transfer learning techniques was trained and tested on both ChEMBL and BindingDB datasets. It can predict the pIC50 values of small molecules that have not been examined experimentally in the ZINC dataset. Additionally, the predicted pIC50 values benefitted the classification of bioactivity  (i.e. active/inactive) by manually setting a threshold. While this threshold is normally 6.0 in many research papers, it is crucial to verify the reliability via statistical analysis, such as p-value analysis.

The aim of LogicRegression classification model was to further improve the reliability of prediction results. If both regression model and classification model identify the untested compounds as active, then it is reliable.

Note: This is a practice for AI Drug Discovery (AIDD) **without** further analysis of biological networks and pathways. Therefore, the conclusion drawn from this practice may be meaningless.

# 2. Usage

The detailed procedures for constructing these models have been specified in the name of each file.

# 3. Conclusion

Both these two models can perform well. Here are some thoughts and potential problems.

## 3.1.  Different distribution of data in both datasets

Such a different distribution leads to deviations in training results. The use of ChEMBL datasets resulted in an R2 score of around 0.72, while the figure for BindingDB datasets was only 0.47. So, I made the judgement based on these results: ChEMBL datasets contained more high-quality data than BindingDB datasets. To deal with this problem, I apply transfer learning for both ChEMBL and BindingDB datasets. The sequence of using these two datasets was determined by trial and error. The results showed that training the model with BindingDB data ended with better overall performance (**around 0.82 of R2 score**). The potential explanation could be that the model learned noises initially, increasing the robustness when it was transferred to less-polluted ChEMBL datasets.

## 3.2 Imbalanced bioactivity labels in datasets

Although using a pIC50 value of 6.0 as a threshold to classify the bioactivity can pass all statistical analysis, the resulting datasets contained much more 'active' data than 'inactive' data. Therefore, the parameter of class_weight must be set to be '**balanced**'.


## 3.3 Optimizing model

The nature of the QSAR model hinders the flexibility of optimization to some extent. Such a model mainly uses 2D structure 'smiles' to store the physical and chemical information, calculate molecular descriptors, (such as fingerprints, molecular weight and logP value), and predict the bioactivity. In other words, the degree of freedom for optimization is limited in 2D structure since much useful information may be lost during this 'compression' from 3D to 2D. While Graph Neural Network (GNN) might help, it is a trade-off strategy because the size of datasets must increase for training.
