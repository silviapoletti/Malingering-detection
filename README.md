# Malingering detection

<a target="_blank" href="https://colab.research.google.com/github/silviapoletti/Malingering-detection/blob/e142b9e687d54b86487ba1b22489956555164966/Malingering_detection_notebook.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<img align="right" width="30%" src="https://api.hub.jhu.edu/factory/sites/default/files/styles/hub_medium/public/lie.jpg">

The Impact of Event Scale Revised (IES-R) is a questionnaire of 22 questions called items, divided into 3 subscales that are representative of the major symptom clusters of the Post Traumatic Stress Disorder (PTSD): Intrusion, Avoidance, Hyperarousal.

The dataset contains 179 healthy controls taking the test twice. Subjects were asked to first respond to the questions honestly - with a score equal to 1, 2, 3, 4 or 5 - and then fake the PTSD symptoms. Therefore, we have to face the phenomena of "fake bad" concerning the exaggeration of a symptom or a disorder in order to obtain some gain or legal advantage.

The project's aims are summarized as follows:
- Discriminate between honest and dishonest subjects;
- Identify which item’s response underwent faking and which not;
- Reconstruct the honest response given the faked ones.

In the first part of the notebook we will compute some exploratory analysis in order to better understand the data we are working with. Then, we extract some peculiar characteristics of the data by means of classification models and faking detection. Finally, we use regression models to reconstruct the honest responses starting from the dishonest ones.

# Exploratory Data Analysis

According to the instructions, the users are expected to use relatively low scores (honest responses) in the first questionnaire and then increment some of these scores in the second questionnaire, in order to fake PTSD sympthoms (dishonest responses). 

The percentage of changed answers in the expected direction is pretty high, about 81%. This means that dishonest subjects don't lie in every single question, but in the majority. Therefore the dataset can be considered relevant for our project's goals.

Items 9, 15 and 19 are in the top-6 lowest average scores of honest subjects and top-6 highest average scores of dishonest subjects, and therefore we could expect that those items will be the most useful to discriminate between honest and dishonest subjects: this is confirmed by looking at the **Density plots** below showing that data for those items are almost linearly saparable. On the contrary, items 4 and 5 show an opposite behavior and therefore are expect to be the most misleading for discriminating between honest and dishonest subjects. This is confirmed by the overlapping density plots.

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/51a723b352778bf52ab7d8c3e4367eb95a0717dd/plots/separable_density_plots.png" width="40%"/>
   <img src="https://github.com/silviapoletti/Malingering-detection/blob/51a723b352778bf52ab7d8c3e4367eb95a0717dd/plots/overlapping_density_plots.png" width="40%"/>
</p>

The honest and dishonest answers are uncorrelated both at item level, because the answers to an item in the honest condition are not correlated with the answers to the same item in the dishonest condition. Indeed, the **Correlation matrix** is as follows:

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/51a723b352778bf52ab7d8c3e4367eb95a0717dd/plots/correlation_matrix.png" width="50%"/>
</p>

The correlation at subject level, i.e. how much the anwers of a subject in the honest condition are correlated with the answers of the same subject in the dishonest condition, is only 0.1. Therefore, we can expect that the reconstruction of the honest subject from the fake ones is a very difficult task.

In the following **Target Plots**, the possible scores that a subject can give to an item are reported in the x-axis. The green bars are the histograms reporting the frequency of a given answer to a given item. The numbers in the dark rectangles are in the range $\[0, 1\]$ and correspond to the conditional probability to be dishonest given a certain answer.

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/f0d8f7bb5f442d91e23bd21aa28b1cef54ad34b4/plots/conditional_expectation_item14.png" width="65%"/>
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/f0d8f7bb5f442d91e23bd21aa28b1cef54ad34b4/plots/conditional_expectation_item5.png" width="65%"/>
</p>

Therefore, the probability of beeing dishonest given a high score to item 14 is very high, as well as the probability of beeing honest given a low score. On the other hand, for item 5, whatever the answer is, the subjects are more or less equally likely to be honest and dishonest.

# Classification between honest and dishonest subjects

We used several classification models, namely:
- **K-Nearest Neighbors** (K-NN)
- **Logistic Regression**
- **XGBoost**
- **Decision Tree**
- **Random Forest**. 

In addition, we used [SHAP](https://shap.readthedocs.io/en/latest/index.html) (SHapley Additive exPlanations), that is an approach derived from game theory to explain the output of any machine learning model, and Partial Dependence Plots to show the marginal effect that one or two features have on the predicted outcome of a machine learning model.

Since our dataset is very small, instead of simply using some fixed train and a test set, we used K-fold cross-validation to evaluate the models, with $K=10$.
  
KNN reaches the best performance of 96% of accuracy. The worst performing model is Decision Tree with an accuracy of 91%, however it is the most interpretable one.

The following plots represent the feature importance ranking (left) and the **SHAP scores** (right) based on the fitted Random Forest model. 

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/ec20102b3cb26eeb549c7a7b346ec239d4ede797/plots/random_forest_ranking.png" width="55%"/>
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/ec20102b3cb26eeb549c7a7b346ec239d4ede797/plots/random_forest_shap.png" width="40%"/>
</p>

The most relevant features for classification are item 9 and 14 both belonging to the Intrusive subscale. This means that the imagery associated with the traumatic event and nightmares is probably decisive for the distinction between honest and dishonest subjects, because liars tend to exaggerate more the symptoms described by this subscale.

For all the items except items 1, 4 and 5, a high score (red) corresponds to a high probability to be classified as dishonest, while a low score (blue) corresponds to a high probability to be classified as honest.
The majority of the items has high absolute values of SHAP, with a more clear and strong distinction between honest and dishonest subjects, resulting in a very good classification accuracy. Therefore, the SHAP value is useful for distinguishing between the two classes quite easily for the majority of the questions.

The following **Partial dependence plot** on the left indicate that a high score to item 9 combined with a high score to item 14 imply a probability higher than 60% for a subject to be classified as dishonest. While a low score in both the items implies the opposite. The other plot on the right shows that item 9 is much more relevant in predicting the dishonest than item 5: we could classify an individual as dishonest with probability higher than 55% if their score in item 9 is 4 or 5, and the score of item 5 would not affect our decision.

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/cb6bcffc2161c017a06c701b767f34e067e6d7be/plots/partial_dependence_9and14.png" weight="70%"\>
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/cb6bcffc2161c017a06c701b767f34e067e6d7be/plots/partial_dependence_9and5.png" weight="70%"\>
</p>

We can also pick some samples in the dataset and apply **SHAP Tree explainer** to understand how each item contributed (with its importance) to the classification of the samples in the Random Forest model.  
Blue items indicate a "positive" influence on the final decision, in the direction of predicting the subject as honest, while red items indicate a "negative" influence on the final decision, in the direction of predicting the subject as dishonest. Note that:
- The bigger the length of the "thick arrows", the greater the importance of the corresponding item in the decision making;
- The closer the "thick arrows" to the prediction, the greater the importance of the corresponding item in the decision making.

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/e3d4249e7967fb7381fff0f5d2fcd103aee839ec/plots/shap_classification.png" \>
</p>

The misclassification is interesting: a low score given in items 1 and 3 is correctly interpreted by the model as an indicator that the subject is dishonest. However all the low values given in the other items give more relevance to the hypothesis that the subject is honest.

Dimensionality reduction with **t-Stochastic Neighbor Embetting** (t_SNE) works pretty well for classification: the new 2D data remains informative despite their dimension and can make the accuracy grow.

Overall, we can see that just one feature (item 9) is enough to reach a satisfactory accuracy, but the best performances are obtained with at least 4 features (chosen among the one having the highest feature importance).

In conclusion, the classification accuracy increases to 96% with the use of four **engineered features**, namely the subject's mean score for the three subclasses and the IES-R score, defined as the sum of all the scores given by the subject.

# Lie detection

We perform anomaly detection to investigate which item response underwent faking and which not. We will approach the problem in the other way around: given a dishonest questionnaire, we'll catch those answers that are honest, i.e. that didn't change in the expected direction from the first questionnaire to the second one. In other words, we will detect the outliers (i.e. honest responses) among the faked answers in the second questionnaire, at item level and subject level.
To find the outliers at item level we will take one specific item and consider all its corresponding answers: those answers that differ very much from the others are considered as outliers and are therefore the honest answers that the subjects gave in the dishonest condition.
To find the outliers at subject level we will take one specific subject and consider all their answers: we then classify each answer of the subject as outlier or not, depending on the usual values that each specific item takes in the dishonest condition.

We consider three strategies:
- **Term Frequency - Inverse Document Frequency** (TF-IDF) score is defined as $$TF\text{-}IDF_{i,j}(s) = TF_{i,j}(s)\times IDF_i(s)$$ where $s$ is the score (taking values $1, 2, 3, 4, 5$) corresponding to item $i$ (taking values $1, \dots, 22$) and subject $j$ (taking values $1, …, 176$). 
  - The TF score is defined as: $$TF_{i,j}(s) = \frac{n_{i,j}(s)}{\text{TOTitems}}$$ where $n_{ij}(s)$ is the number of times the the subject $j$ uses the score $s$ in its answers, normalized by the total number of answers the subject gives ($\text{TOTitems} = 22$).
  - The IDF score determines the weight of rare scores across all answers in the dataset and is defined as: $$IDF_i(s) = log\bigg(\frac{N}{n_i(s)} \bigg)$$ where $N = 176$ is the total number of participants and $n_i(s)$ is the number of times the score $s$ was used by other partecipants to answer item $i$. 
- **Isolation Forest** (IF) is built on the basis of decision trees and aims to explicitly identify anomalies instead of profiling normal data points. In these trees, random partitions are created by first randomly selecting a feature and then selecting a random split value between the minimum and maximum value of the selected feature. Therefore, outliers should be identified closer to the root of the tree with fewer splits necessary than normal points. 
The IF anomaly score is defined as: $$s(x,n) = - 2^{-\frac{E[h(x)]}{c(n)}} \in [-1,0]$$ where $h(x)$ is the path length of observation $x$, $c(n)$ is the average path length of unsuccessful search in a Binary Search Tree and $n$ is the number of external nodes. The decision making is described as follows:
  - A score close to $-1$ indicates anomalies;
  - A score whose absolute value is much smaller than $0.5$ indicates normal observations.
- **TF-IDF revised with Isolation Forest** is a weighted sum of the IF anomaly score and the TF-IDF score: $$\text{CombinedScore} = \alpha \cdot | \text{IFScore} | + (1 - \alpha ) \cdot (1 - \text{TFIDFScore})$ where $\alpha, |\text{IFScore}|, \text{TFIDFScore} \in (0,1).$$ Indeed, what corresponds to a honest response, i.e. an outlier, is a high absolute value of the IF anomaly score and a low value of the TF-IDF score.

The best threshold for computing lie detection with TF-IDF is computed as the percentile score in the TF-IDF values distribution for each IES-R item that maximizes the accuracy while minimizing the number of wrong predictions. Here, the prediction of an outlier is considered accurate if the dishonest answer score given by a subject is lower than their honest answer score.
We first estimate the IDF scores associated to each item considering only the answers from honest subjects, then compute the TF-IDF for all the subjects in the test set and finally determine the outliers according to the best threshold (`thr` in the figure) found.

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/cc0c0ad66d5ae4c7573e5953d897a61fca8a26b4/plots/tfidf_tradeoff.png" width="75%"\>
</p>

As we can see from the plot above, TF-IDF doesn't produce a lot of wrong prediction: for a barely good accuracy of 48.1% we have less than 2 wrong predictions, on average. The best threshold correspond to the $thr=95$ percentile.

According to Isolation Forest, for most of the items including item 9 (on the left), honest responses correspond to the scores 1, 2, 3.
Some clear exceptions are item 4 - *I felt irritable and angry* - and item 5 (on the right) - *I avoided letting myself get upset when I thought about it or was reminded of it*. Indeed liars don't think that PTSD patients would avoid negative feelings about their trauma; moreover irritability and anger are feelings that are common also for healthy subjects who tend to give high grades to that question in the honest questionnaire. The **Outlier regions plot** are based on the IF anomaly score and are as follows.

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/7b76e5ac289594d60e63fb6984b67ba8ce9d054e/plots/outlier_region_9.png" width="45%"\>
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/7b76e5ac289594d60e63fb6984b67ba8ce9d054e/plots/outlier_region_5.png" width="45%"\>
</p>

The default threshold value for Isolation Forest is $0.5$. But, again, the best threshold for computing lie detection with Isolation Forest is computed as the value in $[0,1]$ that maximizes the accuracy while minimizing the number of wrong predictions. We first fit the IF algorithm on a train set of dishonest subjects, then we predict the outliers for all the subjects in the test set and finally determine the outliers according to the best threshold (`p` in the figure) found.

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/7522f36f5d8ec0decb17e31455999f2041c18cad/plots/if_tradeoff.png" width="75%"\>
</p>

The best threshold between accuracy and wrong predictions for the Isolation Forest is $p = 0.55$, just a little bigger than the default one $p = 0.5$. The accuracy is much higher than the one obtained with TF-IDF but the number of wrong predictions are more than twice the ones of TF-IDF with the best percentile:

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/7522f36f5d8ec0decb17e31455999f2041c18cad/plots/if_tfidf_wrong_predictions.png" width="75%"\>
</p>

In conclusion, for the best $threshold=0.47$, the TF-IDF revised with Isolation Forest improves the accuracy of the Isolation Forest from 73.6% to 75.7% and also improves the average number of wrong predictions from 4.1 to 3.8. The following shows the performance of the model for various thresholds (`threshold` in the figure).

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/7522f36f5d8ec0decb17e31455999f2041c18cad/plots/combination_tradeoff.png" width="75%"\>
</p>

# Reconstruction of honest responses given the fake ones

We will perform the reconstruction of the honest answers starting from the dishonest ones. As we have seen in the Exploratory Data Analysis section, this is a very hard task, since it seems that the responses of the same subject in the honest and dishonest conditions are not correlated.
Moreover, the dishonest subjects' responses are more correletad than the one of the honest subjects, meaning that there may be exist a pattern for liars; hovever this correlation is still low and therefore we could argue that the dishonest subjects lie in many different ways and this makes the reconstruction task even more difficult. Indeed, we have seen that depending on the subject, the number of changed answer in the expected direction (exaggerating the symphtoms) varies a lot.

Our approach:
- **Trivial strategy** (baseline): Subtract to each subject’s fake response the average of the difference, across all subjects, between fake and honest responses.
- **Linear regression**
- **Ridge Regression**
- **K-Nearest Neighbors**
- **XGBoost**
- **Denoising Autoencoder**: An aoutoencoder learns an efficient representation of the data, using an unsupervised learning approach. Denoising autoencoders are a specific class of autoencoders that have the aim of reconstructing the original input starting from partially corrupted data. Indeed, we can consider dishonest responses as a "noisy" or "corrupted" version of the corresponding subject's honest responses. The autoencoder is composed of two main parts: an encoder and a decoder. The encoder encodes the input into an internal representation, while the decoder reconstruct the output starting from the internal representation of the input. Since our dataset is pretty small, we used only two layers for the encoding part and two for the deconding part, to try to prevent overfitting of the training data.
- **Denoising with Bernoulli Restricted Boltzmann Machine**: A restricted Boltzmann machine (RBM) is a generative stochastic artificial neural network that can learn a probability distribution over its set of inputs. The first step is to perform the unsupervised training to fit the Bernoulli RBM to our data with the use of persistent contrastive divergence. For implementation requirements, we will need binary data, with scores 1, 2 considered as 0 and scores 3, 4, 5 considered as 1. 
  - The raw predictions are just scores that have to be set as values 1, 2, 3, 4 or 5 according to some specific (finetuned) thresholds, corresponding to: $$\alpha_i * max(pred[subject])$$ where $\alpha_i\in (0,1)$. 
  - After applying the thresholds, we will end to have an intermediate prediction with some values set to 0. We will set those values (corresponding to the answers whose RBM score is above $\small{\alpha_i * max(pred[subject])} \, \,\forall i$, i.e. the highest RBM scores) to 1, 2, 3, 4 or 5 according to the other scores of the subject:
    - if the prediction contains more than 20 values equal to 1, then the missing values are set to 1 as well;
    - if the prediction contains 18-19 values equal to 1, then the missing values are set to 2;
    - if the prediction contains 16-17 values equal to 1, then the missing values are set to 3;
    - if the prediction contains 6-15 values equal to 1, then the missing values are set to 4;
    - if the prediction contains 0-5 values equal to 1, then the missing values are set to 5.

This is a winning strategy because is found on the assumption that a honest subject giving all low-valued answers, would not give few but very high answers, except for misleading questions like itwem 4 or 5. Therefore, as the presence of score 1 decreases, we will set the missing values (the one that are supposed to have the greater score of the subject) to increasing values, from 1 to 5. In this way, we can say that a peculiar characteristic of our model is to capture the sudden peaks while predicting reliable low scores for the majority of the other answers. This strategy leads to one of the highest accuracy value in the reconstruction task and moreover it is based on the plausible assumption that subjects have an intern coherence when answering the questions.

We can conclude that XGBoost is the method which obtained the highest reconstruction accuracy (51%), followed by our customized Denoising RBM (50%), KNN (37%), Denoising Autoencoder (36%), Ridge Regression (35%) and Linear Regression (35%).

The next figure refers to standard errors in the reconstruction. The baseline's predictions have the greatest standard error w.r.t. the groundtruth. XGBoost appears to be the model with the highest accuracy at the reconstruction and the lowest standard error at the prediction, and the overall standard error of all the other models is not very high, almost 0.85.

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/01dd6860c6f9fbf98d9c3040780d6e1092e63cfc/plots/reconstruction_errors.png" width="75%"\>
</p>

From the plot below we can clearly see that XGBoost and the Denoising RBM are the algorithms that reach the highest accuracy in almost all the items. The only exception is item IES-R 5, where a little bit higher accuracy is reached by the Denoising Autoencoder. The black line represents the fraction of subjects who lie in a certain item in our dataset (for example, about 90% of the subject lied when ansering to item 9). More or less, the worst performances occur when the amount of lie among the subject in a certain item is low. This is reasonable, because if the subjects don't lie there's no way for the model to understand the liars behavior and consequently reconstruct their honest responses.

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/01dd6860c6f9fbf98d9c3040780d6e1092e63cfc/plots/reconstruction_lie_amount.png" width="75%"\>
</p>



