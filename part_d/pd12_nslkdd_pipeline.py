"""part_d / pd12_nslkdd_pipeline.py

Full pipeline on the NSL-KDD dataset (improved KDD'99 without redundant records).
Mirrors pd11: download/parse, train/val/test split, CNN/LSTM/GRU/Hybrid/GAN models,
accuracy-loss-ROC plots, confusion matrices, timing, and ANOVA/SHAP/LIME analysis.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import security_utils as su

if __name__ == "__main__":
    su.run_pipeline("nslkdd", kinds=("cnn", "lstm", "gru", "hybrid"), epochs=10)
