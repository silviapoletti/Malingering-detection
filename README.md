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
