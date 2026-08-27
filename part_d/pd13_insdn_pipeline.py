"""part_d / pd13_insdn_pipeline.py

Full pipeline on the INSDN SDN/cloud dataset (CSV flows). Download/parse, train/
val/test split, CNN/LSTM/GRU/Hybrid/GAN models, accuracy-loss-ROC plots, confusion
matrices, timing, and ANOVA/SHAP/LIME interpretability. Falls back to synthetic
data if the CSV cannot be downloaded.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import security_utils as su

if __name__ == "__main__":
    su.run_pipeline("insdn", kinds=("cnn", "lstm", "gru", "hybrid"), epochs=10)
