# Malingering detection

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

Items 9, 15 and 19 are in the top-6 lowest average scores of honest subjects and top-6 highest average scores of dishonest subjects, and therefore we could expect that those items will be the most useful to discriminate between honest and dishonest subjects: this is confirmed by looking at the density plots below showing that data for those items are almost linearly saparable. On the contrary, items 4 and 5 show an opposite behavior and therefore are expect to be the most misleading for discriminating between honest and dishonest subjects. This is confirmed by the overlapping density plots.

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/51a723b352778bf52ab7d8c3e4367eb95a0717dd/plots/separable_density_plots.png" width="40%"/>
   <img src="https://github.com/silviapoletti/Malingering-detection/blob/51a723b352778bf52ab7d8c3e4367eb95a0717dd/plots/overlapping_density_plots.png" width="40%"/>
</p>

The honest and dishonest answers are uncorrelated both at item level, because the anwers to an item in the honest condition are not correlated with the answers to the same item in the dishonest condition. Indeed, the correlation matrix is as follows:

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/51a723b352778bf52ab7d8c3e4367eb95a0717dd/plots/correlation_matrix.png" width="50%"/>
</p>

The correlation at subject level, i.e. how much the anwers of a subject in the honest condition are correlated with the answers of the same subject in the dishonest condition, is only 0.1. Therefore, we can expect that the reconstruction of the honest subject from the fake ones is a very difficult task.

In the following plots, the possible scores that a subject can give to an item are reported in the x-axis. The green bars are the histograms reporting the frequency of a given answer to a given item. The numbers in the dark rectangles are in the range $\[0, 1\]$ and correspond to the conditional probability to be dishonest given a certain answer.

<p align="center">
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/f0d8f7bb5f442d91e23bd21aa28b1cef54ad34b4/plots/conditional_expectation_item14.png" width="50%"/>
  <img src="https://github.com/silviapoletti/Malingering-detection/blob/f0d8f7bb5f442d91e23bd21aa28b1cef54ad34b4/plots/conditional_expectation_item5.png" width="50%"/>
</p>

Therefore, the probability of beeing dishonest given a high score to item 14 is very high, as well as the probability of beeing honest given a low score. On the other hand, for item 5, whatever the answer is, the subjects are more or less equally likely to be honest and dishonest.

* [3 -  Classification](#scrollTo=KIhvkK6UFV0-&line=1&uniqifier=1)
    * [3.1 - K-Nearest Neighbors](#scrollTo=8OWvApzMcrNn&line=1&uniqifier=1)
    * [3.2 - Logistic Regression](#scrollTo=9EuHI1jDOY_s&line=1&uniqifier=1)
    * [3.3 - XGBoost](#scrollTo=VgRCq1xK1ret&line=1&uniqifier=1)
    * [3.4 - Decision Tree](#scrollTo=ccchgQ15fqiD&line=1&uniqifier=1)
    * [3.5 - Random Forest](#scrollTo=uM3cdRfBi_8Y&line=1&uniqifier=1)
      * [3.5.1 - Partial Dependence Plots](#scrollTo=Z9-0mKejmwfM&line=1&uniqifier=1)
    * [3.6 - Model explanation](#scrollTo=PbOvefY8rNy4&line=1&uniqifier=1)
    * [3.7 - Dimensionality reduction and engineered features](#scrollTo=juSh4sXrLf7u&line=2&uniqifier=1)
* [4 - Lie Detection](#scrollTo=OMGpjoJowgdK&line=1&uniqifier=1)
    * [4.1 - TF-IDF](#scrollTo=uGR1ITXybtfG&line=14&uniqifier=1)
      * [4.1.1 - Threshold validation](#scrollTo=hMgdmRtL0gwH&line=1&uniqifier=1)
      * [4.1.2 - Lie detection at item level](#scrollTo=I7TJd1C23c3g&line=1&uniqifier=1)
      * [4.1.3 - Lie detection at subject level](#scrollTo=tt2b1La43x_8&line=1&uniqifier=1)
    * [4.2 - Isolation Forest](#scrollTo=gz5H3zeirlGO)
      * [4.2.1 - Outlier regions](#scrollTo=wz3hOMZmJnEL&line=1&uniqifier=1)
      * [4.2.2 - Threshold validation](#scrollTo=6OfGMkg-MtF2&line=4&uniqifier=1)
      * [4.2.3 - Lie detection at item level](#scrollTo=15we7MR2M4I5&line=1&uniqifier=1)
      * [4.2.4 - Lie detection at subject level](#scrollTo=psIVG6cxM9UW&line=1&uniqifier=1)
    * [4.3 - Comparison between TF-IDF and IF](#scrollTo=s7WE6NQKVVO4&line=1&uniqifier=1)
    * [4.4 - TF-IDF revised with Isolation Forest](#scrollTo=a7o-hG3pizVd&line=13&uniqifier=1)
* [5 - Reconstruction](#scrollTo=w3-09mOOoBaZ&line=1&uniqifier=1)
  * [5.1 - Trivial strategy (baseline)](#scrollTo=qoTDsufu5NBv)
  * [5.2 - Linear Regression](#scrollTo=RC6wtqlc0BDC&line=1&uniqifier=1)
  * [5.3 - Ridge Regression](#scrollTo=LSBHfXKHuuKb&line=1&uniqifier=1)
  * [5.4 - K-Nearest Neighbors](#scrollTo=OEanfL374WTt&line=1&uniqifier=1)
  * [5.5 - XGBoost](#scrollTo=LTMEcsmZZgJU&line=1&uniqifier=1)
  * [5.6 - Denoising with Restricted Boltzmann Machines](#scrollTo=a4_7twm_AA4q)
  * [5.7 - Denoising Autoencoder](#scrollTo=gfU-X2TAf0hy)
  * [5.8 - LSTM Autoencoder](#scrollTo=-uz4a1uD2zqb)
  * [5.9 - Final comparison](#scrollTo=4aAEG4Ak03ca&line=1&uniqifier=1)
* [6 - Conclusions](#scrollTo=I_KM4XL8EZtF)
