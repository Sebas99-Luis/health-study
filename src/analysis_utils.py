from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import stats


def load_health_data(file_path: str | Path) -> pd.DataFrame:
    """

    Load dataset från a CSV file.
    file_path -> Path to the CSV file and returns DataFrame
    containing thr dataset.

    """
    return pd.read_csv(file_path)


def calculate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Räkna numerisk statitiskt för dataset
    args -> data frmae containing study data
    returns -> DF med mean, median, min, max
    """

    columns = ["age", "weight", "height", "systolic_bp", "cholesterol"]

    summary = df[columns].agg(["mean", "median", "min", "max"])

    return summary.round(2)


def calculate_disease_proportion(df: pd.DataFrame) -> float:
    """
    Räkna proportion av deltagare med disease
    """
    proportion = df["disease"].mean()

    return float(proportion)


def simulate_disease(probability: float, n: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Simulation av disease förekomst av deltagarna
        probability -> sannolikhet av har diseas
        n -> nummer av deltagar for simlation
    """
    rng = np.random.default_rng(seed=seed)

    simulated_disease = rng.binomial(n=1, p=probability, size=n)

    return simulated_disease


def calculate_bp_confidence_interval(
    df: pd.DataFrame,
    method: str = "normal",
    confidence_level: float = 0.95,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Räkna confidence interval för mean bp

        df -> DF
        method -> "normal" or "bootstrap"
        confidence_level -> nivå av confindence
        n_bootstrap -> nummer av samples
    """

    bp = df["systolic_bp"].dropna().to_numpy()
    mean_bp = bp.mean()

    if method == "normal":
        std_bp = bp.std(ddof=1)
        standard_error = std_bp / np.sqrt(len(bp))

        # ger oss det kritiska z-värdet för normalfördelning
        z = norm.ppf((1 + confidence_level) / 2)

        lower = mean_bp - z * standard_error
        upper = mean_bp + z * standard_error

    elif method == "bootstrap":
        rng = np.random.default_rng(seed=seed)

        bootstrap_means = np.array(
            [
                rng.choice(bp, size=len(bp), replace=True).mean()
                # använder _ efetersom det bara vill upprepa loopen
                # och inte behöver använda numret på varje repetion
                for _ in range(n_bootstrap)
            ]
        )

        alpha = 1 - confidence_level

        lower, upper = np.quantile(bootstrap_means, [alpha / 2, 1 - alpha / 2])

    else:
        raise ValueError("Metoden måste vara 'normal' or 'bootstrap'")

    return float(lower), float(upper)


def test_smoking_bp(df: pd.DataFrame) -> tuple[float, float]:
    """
    Teste om rökare har högre bp
    """

    smokers_bp = df.loc[df["smoker"] == "Yes", "systolic_bp"].dropna()

    non_smokers_bp = df.loc[df["smoker"] == "No", "systolic_bp"].dropna()

    result = stats.ttest_ind(
        smokers_bp, non_smokers_bp, equal_var=False, alternative="greater"
    )

    return float(result.statistic), float(result.pvalue)
