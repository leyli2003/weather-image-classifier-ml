# Weather Condition Recognition from Images (Classical Machine Learning)

This repository documents a research-oriented study on **weather condition recognition from real-world images** using **classical machine learning methods**. The project focuses on designing and evaluating supervised learning pipelines under **non-controlled imaging conditions**, without relying on deep learning or convolutional neural networks.

## Motivation
Visual recognition in natural settings is challenging due to factors such as **illumination changes, viewpoint variability, background clutter, and intra-class diversity**. This project investigates how far classical machine-learning pipelines can be pushed under these conditions, and what limitations emerge when operating without learned deep visual representations.

## Methodology
An end-to-end supervised learning workflow was designed, covering:
- Data preparation and label handling
- Classical feature extraction from images
- Model training using standardized scikit-learn pipelines
- Comparative evaluation of multiple classifiers
- Serialization of trained models for reproducibility

Multiple models were trained and compared, with attention to robustness and generalization rather than dataset-specific optimization.

## Models
The study evaluates several classical classifiers implemented within unified pipelines, including:
- Support Vector Machines (SVM)
- Multilayer Perceptron (MLP)
- Additional baseline classifiers for comparison

All trained models are exported as reusable pipeline artifacts.

## Tools & Technologies
- Python
- scikit-learn
- joblib

## Repository Structure (planned)
