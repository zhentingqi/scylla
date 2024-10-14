import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import rel_entr


def kl_divergence(p_scores, q_scores, num_bins=100):
    """KL Divergence: How much information is lost when Q is used to approximate P"""
    # Estimate the PDFs using KDE
    p_kde = gaussian_kde(p_scores)
    q_kde = gaussian_kde(q_scores)

    # Define a range of values to evaluate the PDFs
    min_score = min(np.min(p_scores), np.min(q_scores))
    max_score = max(np.max(p_scores), np.max(q_scores))
    x = np.linspace(min_score, max_score, num_bins)

    p_pdf = p_kde(x)
    q_pdf = q_kde(x)

    # Avoid division by zero
    p_pdf = np.clip(p_pdf, 1e-10, None)
    q_pdf = np.clip(q_pdf, 1e-10, None)

    # Calculate KL Divergence
    kl_div = np.sum(rel_entr(p_pdf, q_pdf))

    return kl_div


def js_divergence(p_scores, q_scores, num_bins=100):
    """JS Divergence: A symmetrized and smoothed version of KL Divergence"""
    # Estimate the PDFs using KDE
    p_kde = gaussian_kde(p_scores)
    q_kde = gaussian_kde(q_scores)

    # Define a range of values to evaluate the PDFs
    min_score = min(np.min(p_scores), np.min(q_scores))
    max_score = max(np.max(p_scores), np.max(q_scores))
    x = np.linspace(min_score, max_score, num_bins)

    p_pdf = p_kde(x)
    q_pdf = q_kde(x)

    # Avoid division by zero
    p_pdf = np.clip(p_pdf, 1e-10, None)
    q_pdf = np.clip(q_pdf, 1e-10, None)

    # Calculate JS Divergence
    m = 0.5 * (p_pdf + q_pdf)
    js_div = 0.5 * np.sum(rel_entr(p_pdf, m)) + 0.5 * np.sum(rel_entr(q_pdf, m))

    return js_div
