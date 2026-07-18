import numpy as np
import scipy.stats as stats

def run_wilcoxon_test(scores_a, scores_b, metric_name="Metric"):
    """
    Wilcoxon signed-rank test for comparing paired F1 scores.
    """
    stat, p_value = stats.wilcoxon(scores_a, scores_b)
    print(f"Wilcoxon signed-rank test for {metric_name}: statistic={stat:.4f}, p-value={p_value:.4e}")
    return stat, p_value

def generate_confidence_interval_report(scores, metric_name="Metric", confidence=0.95):
    """
    Generates confidence intervals based on the standard t-distribution.
    """
    scores = np.asarray(scores)
    n = len(scores)
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1) if n > 1 else 0.0
    
    if n <= 1:
        ci = 0.0
    else:
        sem = std_score / np.sqrt(n)
        ci = stats.t.ppf((1 + confidence) / 2.0, df=n-1) * sem
        
    print(f"{metric_name}: Mean = {mean_score:.4f}, Std = {std_score:.4f}, 95% CI = ±{ci:.4f}")
    return mean_score, std_score, ci
