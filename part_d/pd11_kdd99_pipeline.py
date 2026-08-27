"""part_d / pd11_kdd99_pipeline.py

Full pipeline on the KDD'99 dataset: download/extract, parse, split into
train/val/test, train CNN / LSTM / GRU / Hybrid / GAN-augmented models, and report
accuracy, loss, ROC, confusion matrices, time complexity, and ANOVA/SHAP/LIME
interpretability. Falls back to synthetic data if the dataset cannot be downloaded.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import security_utils as su

if __name__ == "__main__":
    su.run_pipeline("kdd99", kinds=("cnn", "lstm", "gru", "hybrid"), epochs=10)
