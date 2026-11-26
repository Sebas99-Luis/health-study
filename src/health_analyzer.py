import pandas as pd
from .analysis_utils import (
    calculate_summary_stats,
    calculate_disease_proportion,
    simulate_disease,
    calculate_bp_confidence_interval,
    test_smoking_bp,
)
import matplotlib.pyplot as plt


class HealthAnalyzer:
    """
    Klass för att analisera health study data

    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summary_statistics(self) -> pd.DataFrame:
        """
        Räkna numerisk statistisk för dataset
        """

        return calculate_summary_stats(self.df)

    def disease_proportion(self) -> float:
        """
        Räkna proporktion av deltagare med deisease
        """
        return calculate_disease_proportion(self.df)

    def disease_simulation(self, n: int = 1000, seed: int = 42) -> float:
        """
        Simularing av disease occurrence med dataset sannolikhet
        """
        probability = self.disease_proportion()

        simulated = simulate_disease(probability=probability, n=n, seed=seed)

        return float(simulated.mean())

    def confidence_interval(
        self,
        method: str = "normal",
        confidence_level: float = 0.95,
        n_bootstrap: int = 5000,
        seed: int = 42,
    ) -> tuple[float, float]:
        """
        Räkna confidene interval föe mean bp"""

        return calculate_bp_confidence_interval(
            self.df,
            method=method,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    def smoking_bp_test(self) -> tuple[float, float]:
        """
        Testa om rökare har högre bp
        """
        return test_smoking_bp(self.df)

    def plot_bp_distribution(self, bins: int = 20):
        """Skapa histogram av bp"""

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(self.df["systolic_bp"].dropna(), bins=bins, edgecolor="black")

        ax.set_title("Distribution av Systolic Blood Pressure")
        ax.set_xlabel("Systolic Blood Pressure")
        ax.set_ylabel("Nummer av deltagare")

        fig.tight_layout()

        return fig

    def plot_weight_by_gender(self):
        """Skapa a boxplot jamföra vikt med kön"""

        # Dela upp deltagarna efter kön
        female_vikt = self.df.loc[self.df["sex"] == "F", "weight"].dropna()

        male_vikt = self.df.loc[self.df["sex"] == "M", "weight"].dropna()

        # Skapa boxplot

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.boxplot(
            [female_vikt, male_vikt],
            tick_labels=["Female (F)", "Male (M)"],
            patch_artist=True,
        )

        ax.set_title("Viktfördelning efter kön ")
        ax.set_xlabel("Kön")
        ax.set_ylabel("Vikt (kg)")

        fig.tight_layout()

        return fig

    def plot_disease_by_sex(self):
        """
        Skapa barchat med disease efter kön
        """

        # Beräkna andelen med sjukdom per kön
        disease_percentage = self.df.groupby("sex")["disease"].mean() * 100

        # Välj ordning på grupperna
        disease_percentage = disease_percentage.reindex(["F", "M"])

        # Skapa stapeldiagram
        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(["Female (F)", "Male (M)"], disease_percentage.values)

        # Visa procent ovanför staplarna
        ax.bar_label(bars, fmt="%.2f%%", padding=3)

        ax.set_title("Disease Prevalence by Sex")
        ax.set_xlabel("Sex")
        ax.set_ylabel("Disease Prevalence (%)")
        ax.set_ylim(0, disease_percentage.max() * 1.2)

        fig.tight_layout()

        return fig

    def plot_cholesterol_vs_bp(self):
        """
        Skapa a scatter plot av kolesterol och bp
        """

        # Beräkna korrelationen
        correlation = self.df["cholesterol"].corr(self.df["systolic_bp"])

        # Skapa scatter plot
        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(self.df["cholesterol"], self.df["systolic_bp"], alpha=0.4)

        ax.set_title("Cholesterol vs Systolic Blood Pressure")
        ax.set_xlabel("Cholesterol (mmol/L)")
        ax.set_ylabel("Systolic Blood Pressure (mmHg)")

        # Visa korrelationen i grafen
        ax.text(
            0.05,
            0.95,
            f"Pearson r = {correlation:.4f}",
            transform=ax.transAxes,
            va="top",
        )

        ax.grid(alpha=0.3)
        fig.tight_layout()

        return fig
