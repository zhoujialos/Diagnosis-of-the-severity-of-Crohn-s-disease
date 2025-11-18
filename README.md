# Few-Shot Learning for Crohn's Disease Severity Classification

## Overview

This project implements and compares **Few-Shot Learning** methods with **Traditional Machine Learning** approaches for Crohn's disease severity classification. The implementation includes three few-shot learning architectures (Metric Learning, Matching Networks, and Prototypical Networks) and five traditional ML methods (SVM, Random Forest, Decision Tree, XGBoost, LightGBM).

## Workflow

![Medical Flow](medical%20flow.png)

## Project Structure

```
few_shot_paper/
├── few_mutil_features.py      # Main implementation file
├── PerClassInspector.py        # Per-class analysis and visualization tool
├── train_shap.csv              # Training dataset
├── test_shap.csv               # Test dataset
├── best_params_*.json          # Best hyperparameters for each model
├── *.csv                       # Experiment results and configurations
└── README.md                   # This file
```

## Key Features

### 1. Few-Shot Learning Methods
- **Metric Learning**: Learns embeddings using triplet loss with a classifier head
- **Matching Networks**: Uses attention-based classification with cosine similarity
- **Prototypical Networks**: Classifies based on distance to class prototypes

### 2. Traditional Machine Learning Methods
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree
- XGBoost
- LightGBM

### 3. Data Processing
- **Text Feature Extraction**: Binary (0/1) encoding of medical text phrases
- **Categorical Features**: CDAI score, SESCD score, FC, gender, smoking, education
- **Continuous Features**: CRP, age (standardized)
- **Leak-free Cross-Validation**: Independent feature extraction per fold

### 4. Hyperparameter Optimization
- Bayesian optimization using Optuna
- 5-fold stratified cross-validation
- Median pruning for efficient search
- 100 trials per model

### 5. Comprehensive Evaluation
- Per-class accuracy and F1 scores
- Confusion matrices
- Embedding visualizations (UMAP/t-SNE)
- Intra-class vs inter-class distance analysis
- Top-k accuracy metrics
- Support-query similarity heatmaps (for Matching Networks)

## Requirements

```bash
# Core dependencies
numpy
pandas
scikit-learn
torch
pytorch-metric-learning
optuna

# Traditional ML
xgboost
lightgbm

# Visualization
matplotlib
umap-learn  # Optional, falls back to t-SNE
```

## Installation

```bash
# Install dependencies
pip install numpy pandas scikit-learn torch pytorch-metric-learning optuna xgboost lightgbm matplotlib

# Optional: Install UMAP for better visualizations
pip install umap-learn
```

## Usage

### Basic Usage

Simply run the main script:

```bash
python few_mutil_features.py
```

This will:
1. Load training and test data
2. Optimize all 8 methods (3 few-shot + 5 traditional ML)
3. Train final models with best hyperparameters
4. Evaluate on test set
5. Generate visualizations and save results

### Configuration

Modify the `ComparisonConfig` class to adjust settings:

```python
class ComparisonConfig:
    # Data paths
    TRAIN_DATA_PATH = 'train_shap.csv'
    TEST_DATA_PATH = 'test_shap.csv'
    
    # Cross-validation
    N_SPLITS = 5
    RANDOM_STATE = 42
    
    # Optimization
    N_TRIALS = 100
    
    # Few-Shot Learning
    N_WAY = 3          # 3-way classification
    N_SUPPORT = 5      # 5 support samples per class
    N_QUERY = 3        # 3 query samples per class
```

### Custom Vocabulary

Update the `VOCABULARY` dictionary in `ComparisonConfig` to add/modify medical text features:

```python
VOCABULARY = {
    "thickening and narrowing of the small intestine": 0,
    "thickening and narrowing of the colon": 1,
    # ... add more features
}
```

## Output Files

### JSON Files (Best Parameters)
- `best_params_metric_learning.json`
- `best_params_matching_network.json`
- `best_params_prototypical_network.json`
- `best_params_svm.json`
- `best_params_random_forest.json`
- `best_params_decision_tree.json`
- `best_params_xgboost.json`
- `best_params_lightgbm.json`

### CSV Files (Results)
- `model_performance_summary.csv`: Overall performance comparison
- `cv_fold_scores.csv`: Detailed cross-validation scores
- `experiment_config.csv`: Experiment configuration
- `best_method_info.csv`: Best performing method
- `method_ranking.csv`: Methods ranked by performance

### Visualization Files (in `figs/inspect/<model_name>/`)
- `confmat.png`: Confusion matrix
- `intra_inter_hist.png`: Distance distribution histogram
- `prototype_distance_heatmap.png`: Distance to class prototypes
- `embedding_umap.png` or `embedding_tsne.png`: 2D embedding visualization
- `topk_by_class.png`: Top-k accuracy per class
- `matching_support_query_similarity.png`: Support-query similarity (Matching Networks only)

## Architecture Details

### Shared Backbone

All three few-shot learning methods share the same feature extractor:

```
Input → Linear(hidden_size) → ReLU → Dropout → 
Linear(hidden_size//2) → ReLU → Dropout → 
Linear(embedding_size) → ReLU → Embeddings
```

### Metric Learning
- **Loss**: Triplet Margin Loss + Cross-Entropy Loss
- **Evaluation**: Linear probe (Logistic Regression on embeddings)

### Matching Networks
- **Similarity**: Cosine similarity between support and query embeddings
- **Classification**: Attention-based weighted aggregation
- **Optional**: Full Context Embeddings (FCE) using Bidirectional LSTM

### Prototypical Networks
- **Prototypes**: Mean embedding of support samples per class
- **Classification**: Negative Euclidean distance to prototypes

## Data Format

### Input CSV Format

```csv
text,CRP,age,CDAI_score,SESCD_score,FC,gender,smoking,education,label
"thickening and narrowing of the small intestine, colon fistula",15.2,45,280,12,500,1,0,2,2
...
```

### Features
- **text**: Medical findings (comma-separated phrases)
- **CRP**: C-reactive protein level
- **age**: Patient age
- **CDAI_score**: Crohn's Disease Activity Index
- **SESCD_score**: Simple Endoscopic Score for Crohn's Disease
- **FC**: Fecal Calprotectin
- **gender**: 0=Female, 1=Male
- **smoking**: 0=No, 1=Yes
- **education**: Education level (ordinal)
- **label**: Disease severity class (0, 1, 2)

## Evaluation Metrics

1. **Accuracy**: Overall classification accuracy
2. **F1 Score**: Weighted F1 score across all classes
3. **Per-class Metrics**: Precision, recall, F1 for each severity level
4. **Cross-validation Statistics**: Mean ± standard deviation across folds

## Reproducibility

All experiments use fixed random seeds for reproducibility:

```python
set_all_seeds(42)  # Sets seeds for Python, NumPy, PyTorch, and CUDA
```

## Advanced Usage

### Using PerClassInspector

```python
from PerClassInspector import PerClassInspector

# After training a model
inspector = PerClassInspector(
    trainer=trainer,
    trained=final_model,
    class_names=['Mild', 'Moderate', 'Severe'],
    model_name='metric_learning',
    random_state=42
)

# Generate all visualizations
inspector.run_all(
    save_dir='figs/inspect/my_model',
    distance_metric='cosine',
    embed_vis='umap',
    do_matching_heatmap=True,
    topk_list=[1, 3, 5]
)
```

### Custom Model Training

```python
# Create trainer
trainer = UnifiedTrainer(
    model_type='prototypical_network',
    train_features=train_features,
    train_labels=train_labels,
    test_features=test_features,
    test_labels=test_labels,
    train_df=train_data,
    base_processor=processor
)

# Optimize hyperparameters
study = trainer.optimize()

# Train final model
best_params = study.best_params
final_model = trainer.train_final_model(best_params)

# Evaluate
test_acc, test_f1 = trainer.evaluate_few_shot_samplewise(final_model)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Few-Shot Prototypical Networks Enable Computed Tomography Enterography-Independent, High-AUC Severity Stratification of Crohn’s Disease at First Diagnosis},
  author={JiaLi Zhou},
  year={2025}
}
```


## Contact

[Your contact information]

## Acknowledgments

- PyTorch Metric Learning library for metric learning implementations
- Optuna for hyperparameter optimization
- scikit-learn for traditional ML methods and evaluation metrics

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in `ComparisonConfig` or use CPU:
```python
DEVICE = 'cpu'
```

### Slow Training
Reduce number of trials or episodes:
```python
N_TRIALS
num_episodes  # In trial suggestions
```

### Missing UMAP
Install with: `pip install umap-learn` or the code will automatically fall back to t-SNE.


## Version History

- **v1.0** (2025): Initial release with 3 few-shot methods and 5 traditional ML methods
