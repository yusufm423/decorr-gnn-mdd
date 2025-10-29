# Enhancing Depression Diagnosis with Augmented Brain Signal Driven Decorrelated Graph Neural Networks

## Abstract
**Background:** Major Depressive Disorder (MDD) is a leading global neuropsychiatric disorder, requiring precise diagnosis for effective intervention. Developing accurate diagnostic models for MDD remains a critical but challenging task. This study introduces a graph-based deep learning framework that addresses the issue of limited training data and facilitates robust training for identifying MDD across diverse episode patterns.


**Method:** We introduce Brain Augmented-Decorrelated Network (BrainADNet), a framework designed to address data scarcity by augmenting brain signal inputs. BrainADNet builds upon the Skip-Graph Convolutional Network to aggregate informative multi-layer features, enriching its representational capacity. Recognizing the clinical relevance of demographic factors such as age, education, and gender in depression, we incorporate these attributes into the training process and examine their effect on diagnosis. To further improve feature diversity and reduce overfitting, we use a decorrelation regularizer to the model training. This encourages GCN embeddings to learn complementary, non-redundant representations from input graphs.


**Results:** The framework surpasses all existing models in accurately identifying MDD cases across different depressive stages. We present a detailed ablation study, demonstrating the contribution of each component of our framework to diagnostic precision. Our study highlights top-10 brain regions influential for diagnosing MDD patients in both the genders, addressing a critical gap in understanding gender-specific neural mechanisms. We also uncover distinct patterns in latent-space brain connectivity, derived from GCN embeddings, between individuals experiencing single versus multiple episodes of depression.


**Conclusions:** This study underscores the potential of graph methods to advance diagnostic precision for MDD. By integrating gender-specific and stage-wise insights, our framework equips medical professionals and researchers to design personalized and targeted therapeutic strategies, offering transf
ormative implications for patient care.

<img width="16384" height="6290" alt="BrainADNet_framework" src="https://github.com/user-attachments/assets/fd8f166d-e16c-477d-9053-bb1c7baf07ca" />
