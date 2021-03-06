# Malingering detection

The Post Traumatic Stress Disorder (PTSD) is a typical symptomatic response to the exposure to traumatic life events. 
The Impact of Event Scale (Horowitz, 1979) is a 22-item self-report measure that assesses subjective distress caused by traumatic events. According to this assessment measure, the core characteristics of the PSTD is the distressing oscillation between intrusion and avoidance. 
The IES-R (Weiss & Marmar, 1997) is a revised version of the Impact of Event Scale and was developed as the original version did not include a hyperarousal subscale. 
Therefore the IES-R questionnaire is divided into 3 subscales that are representative of the major symptom clusters of PTSD: Intrusion, Avoidance, Hyperarousal.

The dataset contains 179 healthy controls taking the test twice. Subjects were asked to first respond honestly and then fake the PTSD symptoms. Therefore, we have to face the phenomena of "fake bad" concerning the exaggeration of a symptom or a disorder in order to obtain some gain or legal advantage.

The project's aims are summarized as follows:
- Discriminate between honest and dishonest subjects 
- Identify which item’s response underwent faking and which not
-	Reconstruct the honest response given the faked ones

In the first part of the notebook we will compute some exploratory analysis in order to better understand the data we are working with.Then, we extract some peculiar characteristics of the data by means of classification models and faking detection. Finally, we use regression models to reconstruct the honest responses starting from the dishonest ones.

Table of contents:
# Table of Contents

* [1 - Introduction](#scrollTo=Y1HVqCO_hbqh&line=1&uniqifier=1)
    * [1.1 - IES-R Scale](#scrollTo=fb7z2zD2mAje&line=1&uniqifier=1)
    * [1.2 - Materials and goals](#scrollTo=6kEPTvVme5Zw&line=1&uniqifier=1)
* [2 - Exploratory Data Analysis](#scrollTo=mbwpYSfTdb7S&line=2&uniqifier=1)
    * [2.1 - Changed answers in the expected direction](#scrollTo=6bG379Dk8Et1&line=1&uniqifier=1)
    * [2.2 - Data visualization](#scrollTo=B8u972Lc4QB4&line=1&uniqifier=1)
    * [2.3 - Data correlation](#scrollTo=gwvwTnN37xrG&line=1&uniqifier=1)
    * [2.4 - Conditional expectation](#scrollTo=5Zd2QGGAtvZB&line=1&uniqifier=1)
    * [2.5 - IES-R Subscales](#scrollTo=n9WTS7bZlURR&line=1&uniqifier=1)
      * [2.5.1 - Changed answers in the expected direction](#scrollTo=fkDtMR0AiVvh&line=1&uniqifier=1)
      * [2.5.2 - Data visualization](#scrollTo=oig5gIvS56n4&line=1&uniqifier=1)
      * [2.5.3 - Data correlation](#scrollTo=yDKLRaTX67SU&line=1&uniqifier=1)
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
