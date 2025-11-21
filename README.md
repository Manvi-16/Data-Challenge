# DA5401 – Data Analytics Lab
# Data-Challenge
Predicting LLM Judge Fitness Scores Using Metric Learning

This repository contains the full implementation, experiments, and report for the DA5401 data challenge. The goal of the project is to predict a numerical fitness score between 0 and 10 for a given prompt–response pair under a specific evaluation metric. The score represents how well the model’s response satisfies a criterion such as rejection rate or cultural sensitivity. The score is originally assigned by an LLM judge, and the objective is to approximate that judge’s scoring behaviour.

---

## PROJECT OVERVIEW

Each sample in the dataset contains:

1. A system prompt
2. A user prompt
3. The model’s response
4. A metric name
5. A score between 0 and 10 (training data only)

During testing, the score is not provided and must be predicted.

An important element of the problem is that the metric descriptions are not included as text. Instead, they are provided as fixed 768-dimensional embeddings generated using a Gemma-based embedding model. These must be used as-is.

The main task is therefore to learn how semantically aligned a prompt–response pair is with a given metric name.

---

## DATA PREPROCESSING

The data is multilingual with large portions in Hindi, English, Tamil, Bengali and others. Several preprocessing steps were performed:

1. All missing text fields were replaced with empty strings.
2. A unified format was created by concatenating system prompt, user prompt and response into a single combined query.
3. The combined text was embedded using the sentence-transformers / all-mpnet-base-v2 model.
4. Metric embeddings were loaded from the provided metric_name_embeddings numpy file.
5. Each metric name was mapped to its correct embedding index.
6. All embeddings were saved to avoid recomputation during experimentation.

Simple descriptive analysis was also conducted. Prompts are generally short, while responses vary widely in length. Score distribution is heavily skewed towards values 8 to 10. This skew later motivated the use of inverse-frequency sample weighting.

---

## BASELINE MODELS AND EXPERIMENTATION

Before moving to deep learning, several classical models were tested to build intuition about the structure of the data.

The experiments included:

* Linear regression
* Ridge regression
* PCA combined with regression
* MLP regressor
* Random Forest
* CatBoost
* XGBoost

To support these models, a hand-crafted feature matrix was created consisting of the two embeddings plus a variety of similarity measures such as Euclidean distance, cosine similarity, dot product, Pearson correlation, vector norms and geometric angles.

Ridge regression and XGBoost performed the strongest among these classical baselines. However, all classical models plateaued in terms of RMSE and could not fully capture the relationship between text embeddings and metric embeddings. This motivated the move toward more structured modelling.

---

## LIGHTGBM AND RELATED EXPERIMENTS

Next, LightGBM was tested with several enhancements:

1. Inverse-frequency sample weighting
   Less frequent scores received higher weights to counteract the heavily skewed score distribution.

2. Five-fold cross validation
   This improved stability and produced robust out-of-fold predictions.

3. Blending LightGBM with Ridge
   Different blending weights were tested, but improvements were minimal.

4. Stacking
   Ridge regression was used as a second-level model on top of LightGBM predictions.
   Gains were again marginal.

Overall, LightGBM clearly outperformed the classical baselines, but even after tuning and stacking, performance did not reach the level needed. This suggested the task required a model that could directly learn the joint relationship between metric embeddings and text embeddings.

---

## FINAL MODEL: DUAL ENCODER METRIC LEARNING SYSTEM

The final and best-performing architecture designed for this project was a dual-encoder model with contrastive learning and a regression head.

The model consists of:

1. Text embeddings
   Combined prompt and response text embedded using all-mpnet-base-v2.

2. Metric embeddings
   Fixed 768-dimensional vectors provided in the dataset.

3. Two projection towers
   One for metric embeddings and one for text embeddings.
   Both project into a shared 128-dimensional space.

4. Soft LayerNorm
   Applied before projection for more stable training.

5. Learnable temperature scaling
   Adjusts the sharpness of similarity scores.

6. Contrastive negative sampling
   Each sample pairs with its correct metric and several incorrect (negative) metrics. This teaches the model to distinguish relevant metrics from irrelevant ones.

7. Regression head
   Converts similarity into a calibrated numeric score using a small MLP.

8. Hybrid loss function
   Combines mean squared error, binary cross entropy for positive and negative similarities, and a margin ranking loss.

This architecture was able to learn both semantic alignment and numeric scoring behaviour. It produced the best RMSE across all experiments and aligned well with the judge’s scoring distribution.

---

## RESULT VISUALIZATION

A histogram of the final model’s predicted score distribution is included in the repository. The distribution largely reflects the real-world score skew and shows that the model has learned the judge’s pattern of assigning high scores to most responses.

---


If you want, I can generate a simple requirements.txt or a LICENSE file for your GitHub repo as well.
