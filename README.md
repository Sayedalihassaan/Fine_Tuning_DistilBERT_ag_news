Fine-Tuning DistilBERT for Text Classification on AG News Dataset
This repository contains the code and documentation for fine-tuning a DistilBERT model for text classification on the AG News dataset. The project classifies news articles into four categories: World, Sports, Business, and Science/Tech. The workflow follows best practices in natural language processing (NLP), including data exploration, feature extraction, model fine-tuning, evaluation, and deployment.
Table of Contents

Project Overview
Features
Installation
Usage
Project Structure
Dataset
Methodology
Results
Future Improvements
Contributing
License
Acknowledgements

Project Overview
The goal of this project is to fine-tune a pre-trained DistilBERT model for multi-class text classification using the AG News dataset. The project demonstrates a complete NLP pipeline, from data loading and exploration to model deployment on the Hugging Face Model Hub. Key steps include feature extraction, dimensionality reduction for visualization, model training using Hugging Face's Trainer API, and evaluation with metrics like accuracy and F1-score.
Features

Data Exploration: Visualizes class distribution and decodes labels for interpretability.
Feature Extraction: Extracts [CLS] token embeddings from DistilBERT for downstream tasks.
Dimensionality Reduction: Uses UMAP to visualize high-dimensional embeddings in 2D.
Model Fine-Tuning: Fine-tunes DistilBERT using Hugging Face's Trainer API.
Evaluation: Computes accuracy, weighted F1-score, and visualizes a confusion matrix.
Deployment: Pushes the fine-tuned model to the Hugging Face Model Hub and creates a text classification pipeline.
Best Practices: Ensures reproducibility, GPU acceleration, and comprehensive evaluation.

Installation
To run this project locally, follow these steps:

Clone the Repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:The required libraries are listed in requirements.txt. Install them using:
pip install -r requirements.txt

The requirements.txt includes:
transformers==4.44.0
datasets==2.21.0
matplotlib==3.9.2
umap-learn==0.5.6
accelerate>=0.20.1


Optional: GPU Support:Ensure you have a CUDA-compatible GPU and install PyTorch with CUDA support if needed:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


Hugging Face Hub Login (for model deployment):Run the following command and provide your Hugging Face token:
huggingface-cli login



Usage

Run the Notebook:Open the Fine_Tuning_DistilBERT_ag_news.ipynb notebook in Jupyter or Google Colab:
jupyter notebook Fine_Tuning_DistilBERT_ag_news.ipynb

Follow the cells to execute the pipeline step-by-step.

Inference with the Fine-Tuned Model:After fine-tuning, use the provided pipeline for inference:
from transformers import pipeline
classifier = pipeline("text-classification", model="your-huggingface-username/your-model-name")
result = classifier("Your news article text here")
print(result)


Explore Visualizations:The notebook generates plots for class distribution, UMAP embeddings, and the confusion matrix. These are saved in the notebook output or can be exported as images.


Project Structure
your-repo-name/
├── Fine_Tuning_DistilBERT_ag_news.ipynb  # Main notebook with the project code
├── requirements.txt                      # Dependencies for the project
├── README.md                             # Project documentation
└── LICENSE                               # License file (e.g., MIT)

Dataset
The AG News dataset is sourced from Hugging Face's datasets library (fancyzhx/ag_news). It contains:

Training Set: 120,000 news articles.
Test Set: 7,600 news articles.
Classes: 4 (World, Sports, Business, Science/Tech), balanced with 30,000 samples per class in the training set.

Methodology
The project follows a structured NLP pipeline:

Data Loading: Loads the AG News dataset and decodes labels for interpretability.
Exploration: Visualizes class frequencies using a horizontal bar chart.
Feature Extraction: Uses DistilBERT's [CLS] token embeddings as features.
Dimensionality Reduction: Applies UMAP to visualize embeddings in 2D with hexagonal binning plots.
Fine-Tuning: Trains DistilBERT with a classification head using the Trainer API.
Evaluation: Computes accuracy, weighted F1-score, and visualizes a confusion matrix.
Deployment: Pushes the model to the Hugging Face Model Hub and creates a pipeline for inference.

Tools and Libraries



Library
Purpose



datasets
Dataset loading and management


transformers
Pre-trained models and tokenizers


torch
Tensor operations and GPU acceleration


sklearn
Preprocessing, metrics, and evaluation


umap-learn
Dimensionality reduction


matplotlib
Data visualization


huggingface_hub
Model deployment


Results

Model Performance: The fine-tuned DistilBERT model achieves high accuracy and F1-score on the validation set (exact metrics depend on training configuration).
Visualizations:
Class distribution plot confirms a balanced dataset.
UMAP visualizations show separable clusters for each class.
Confusion matrix highlights areas of misclassification for further analysis.


Deployment: The model is available on the Hugging Face Model Hub for public use.

Future Improvements

Hyperparameter Tuning: Use grid search or Bayesian optimization for optimal parameters.
Advanced Models: Experiment with larger models like BERT or RoBERTa.
Error Analysis: Analyze misclassified examples to improve model robustness.
Data Augmentation: Apply techniques like back-translation or synonym replacement.
Cross-Domain Testing: Evaluate the model on out-of-domain datasets.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please ensure your code follows the project's coding style and includes relevant tests.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements

Hugging Face: For providing the transformers and datasets libraries.
AG News Dataset: For the publicly available dataset.
UMAP: For enabling high-quality dimensionality reduction and visualization.


Feel free to star ⭐ this repository if you find it useful! For questions or feedback, open an issue or contact me at [saiedhassaan2@gmail.com].
