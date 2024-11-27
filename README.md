# Clustering Assignment

This repository contains Colab notebooks demonstrating various clustering algorithms and techniques. Implementations range from scratch-based methods to state-of-the-art embeddings using deep learning models. Each notebook includes proper documentation, visualizations, and quality measures for clustering.

---

## Table of Contents
1. [K-Means Clustering from Scratch]
2. [Hierarchical Clustering]
3. [Gaussian Mixture Models Clustering]
4. [DBSCAN Clustering Using PyCaret]
5. [Anomaly Detection Using PyOD]
6. [Clustering of Time Series Data]
7. [Document Clustering with LLM Embeddings]
8. [Clustering Images Using ImageBind]
9. [Audio Clustering Using ImageBind]

---

## 1. K-Means Clustering from Scratch
**Notebook**: `https://github.com/intimanjunath/Clustering/blob/main/a_K_Means_clustering.ipynb`  
This notebook implements the K-Means clustering algorithm from scratch, detailing the steps of centroid initialization, iterative cluster assignment, and centroid updates. Visualizations and evaluation metrics like inertia are included.

---

## 2. Hierarchical Clustering (Not from Scratch)
**Notebook**: `https://github.com/intimanjunath/Clustering/blob/main/b_Hierarchical_Clustering.ipynb`  
This notebook demonstrates hierarchical clustering using Python libraries. It visualizes the merging of clusters with dendrograms and explains how to interpret hierarchical trees for clustering.

---

## 3. Gaussian Mixture Models Clustering (Not from Scratch)
**Notebook**: `https://github.com/intimanjunath/Clustering/blob/main/c_Gaussian_Mixture_Models_Clustering.ipynb`  
This notebook explains Gaussian Mixture Models (GMMs) for clustering, showcasing how probabilistic cluster assignments work. It includes evaluations using AIC, BIC, and other clustering metrics.

---

## 4. DBSCAN Clustering Using PyCaret
**Notebook**: `https://github.com/intimanjunath/Clustering/blob/main/d_DB_Scan_clustering_.ipynb`  
This notebook uses PyCaret to implement the DBSCAN clustering algorithm, highlighting its density-based approach to identify clusters of varying shapes and noise. It evaluates clustering quality with metrics like silhouette scores.

---

## 5. Anomaly Detection Using PyOD
**Notebook**: `https://github.com/intimanjunath/Clustering/blob/main/e_anomaly_detection_using_pyOD.ipynb`  
This notebook demonstrates anomaly detection using the PyOD library, applying it to univariate and multivariate datasets. Anomalies are detected in time series data, and results are visualized on plots for interpretation.

---

## 6. Clustering of Time Series Data Using Pretrained Models
**Notebook**: `https://github.com/intimanjunath/Clustering/blob/main/f_clustering_of_timeseries_data.ipynb`  
This notebook clusters time series data using pretrained models, demonstrating tasks like stock market analysis. It compares various methods such as TS-PTMs and LLM embeddings for clustering temporal patterns.

---

## 7. Document Clustering with LLM Embeddings
**Notebook**: `https://github.com/intimanjunath/Clustering/blob/main/g_clustering__of_documents.ipynb`  
This notebook illustrates clustering of documents using state-of-the-art embeddings like Sentence Transformers. Semantic similarity is used for clustering, and silhouette scores evaluate the quality of the clusters.

---

## 8. Clustering Images Using ImageBind LLM Embeddings
**Notebook**: `https://github.com/intimanjunath/Clustering/blob/main/h_clustering_with_images.ipynb`  
This notebook performs clustering of images using embeddings from Meta's ImageBind model. It also explores cross-modality embeddings, clustering images alongside other modalities, with visualizations like t-SNE.

---

## 9. Audio Clustering Using ImageBind LLM Embeddings
**Notebook**: `https://github.com/intimanjunath/Clustering/blob/main/i_audio_embeddings.ipynb`  
This notebook clusters audio files using embeddings from the ImageBind model. It applies K-Means clustering and visualizes results with t-SNE and waveform plots.

---

## Common Evaluation Metrics
- **Inertia and Silhouette Scores**: Evaluate the clustering performance.
- **Visualization Techniques**: t-SNE, PCA, dendrograms, and heatmaps for interpreting results.
- **Probabilistic Measures**: Used in GMMs and anomaly detection to assess cluster likelihoods and anomalies.

---

## How to Run
1. Open the desired notebook on Google Colab.
2. Follow the instructions in the notebook to run each cell sequentially.
3. Modify or extend the code as needed to explore additional datasets or configurations.

---

Feel free to raise issues or contribute enhancements via pull requests!
