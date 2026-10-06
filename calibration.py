import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

def compute_ece(y_true, y_probs, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    y_true: array-like of binary ground truth (0 or 1)
    y_probs: array-like of predicted probabilities in [0, 1]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_probs = np.asarray(y_probs, dtype=float)
    
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    if total_samples == 0:
        return 0.0
        
    bin_details = []
    for i in range(n_bins):
        low, high = bin_limits[i], bin_limits[i+1]
        # Include right edge for last bin
        if i == n_bins - 1:
            idx = (y_probs >= low) & (y_probs <= high)
        else:
            idx = (y_probs >= low) & (y_probs < high)
            
        bin_count = np.count_nonzero(idx)
        if bin_count > 0:
            bin_acc = np.mean(y_true[idx])
            bin_conf = np.mean(y_probs[idx])
            bin_err = abs(bin_acc - bin_conf)
            weight = bin_count / total_samples
            ece += weight * bin_err
            bin_details.append({
                "bin": (round(low, 2), round(high, 2)),
                "count": int(bin_count),
                "accuracy": round(float(bin_acc), 4),
                "confidence": round(float(bin_conf), 4),
                "error": round(float(bin_err), 4)
            })
            
    return float(ece), bin_details

def compute_brier(y_true, y_probs):
    y_true = np.asarray(y_true, dtype=float)
    y_probs = np.asarray(y_probs, dtype=float)
    if len(y_true) == 0:
        return 0.0
    return float(brier_score_loss(y_true, y_probs))

class PlattCalibrator:
    """
    Platt Scaling (Logistic Calibration):
    Fits sigmoid: P(Y=1|s) = 1 / (1 + exp(-(a * s + b)))
    where s is the raw detector score.
    """
    def __init__(self):
        self.clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        self.fitted = False
        
    def fit(self, scores, labels):
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        labels = np.asarray(labels, dtype=int)
        
        # Check if we have both classes
        if len(np.unique(labels)) < 2:
            self.fitted = False
            return self
            
        self.clf.fit(scores, labels)
        self.fitted = True
        return self
        
    def predict_proba(self, scores):
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        if not self.fitted:
            # Fallback identity sigmoid if not fitted
            return np.clip(scores.flatten(), 0.0, 1.0)
        return self.clf.predict_proba(scores)[:, 1]
        
    def calibrate_scalar(self, score):
        if not self.fitted:
            return float(np.clip(score, 0.0, 1.0))
        p = self.clf.predict_proba([[float(score)]])[0, 1]
        return float(np.clip(p, 0.0, 1.0))
