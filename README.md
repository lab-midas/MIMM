# Mutual Information Minimization Model (MIMM)


<p align="center">
<img src="./figures/Fig3_MIMM.svg">
</p>
Deep learning models are increasingly being  used in detecting patterns and correlations in medical imaging data such as magnetic resonance imaging. However, conventional methods are incapable of considering the real underlying causal relationships. In the presence of confounders, spurious correlations between data, imaging process, content and output can occur that allow the network to learn shortcuts instead of the desired causal relationship. This effect is even more prominent in new environments or when using out-of-distribution data, since the learning process is primarily focused on correlations and patterns within the data. Hence, wrong conclusions or false diagnoses can be obtained from such confounded models. In this paper, we propose a novel framework,  denoted as Mutual Information Minimization Model (MIMM), that predicts the desired causal outcome while simultaneously reducing the influence of present spurious correlations. The input imaging data is encoded into a feature vector that is split in two components to predict the primary task and the presumed spuriously correlated factor separately. We hypothesize that a learned mutual information between both feature vector components can be reduced to achieve independence, i.e. confounder-free task prediction. The proposed approach is investigated on five databases: two non-medical benchmark databases (Morpho-MNIST and Fashion-MNIST) to verify the hypothesis and three medical databases (German National Cohort, UK Biobank and ADNI). The results show that our model can reduce the mutual information between the feature vector components and thereby, remove the shortcut of the spurious correlation and learn a causal predictive model. Hence, our proposed framework serves as a solution to address the limitations of conventional deep learning models. By explicitly considering and minimizing spurious correlations, it learns causal relationships, leading to more accurate and reliable predictions especially in medical imaging analysis. 

# Prerequisits



```
pip install -r requirements.txt
```

# Running
```
python main.py config.yml
```

# Citing

For citing our work, please use the following bibtex entry:

TBD