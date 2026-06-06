### Stochastic Outer-Layer ###
# Correlated fiscal-risk simulation with portfolio diversification analysis.
#
# Outputs:
#   - correlated_risk_simulation_full.csv
#   - pfram_project_payment_paths.csv
#   - correlated_risk_portfolio_paths.png / .svg
#   - simulated_portfolio_paths_spaghetti.png / .svg
#   - diversification_tail_metrics.csv
#   - diversification_loss_distribution.png / .svg
#   - diversification_tail_metrics.png / .svg

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


np.random.seed(42)

N_SIMS = 10000
YEARS = list(range(2030, 2050))   # extend to contract termination
SHOCK_YEARS = [2030, 2031, 2032, 2033, 2034]

BASE_DEMAND_SHOCK = 0.05
BASE_COST_SHOCK = 0.03
BASE_FINANCING_SHOCK = 0.015

corr = np.array([
    [1.00, 0.00, 0.35],
    [0.00, 1.00, 0.45],
    [0.35, 0.45, 1.00],
])

L = np.linalg.cholesky(corr)

# Replace these with exported PFRAM baseline values
baseline_revenue_b = {
    year: 130 * (1.025 ** (year - 2030))
    for year in YEARS
}

baseline_availability_payment_c = {
    year: 65 * (1.025 ** (year - 2030))
    for year in YEARS
}

# Diversifying project:
# Modelled as a social-housing availability PPP.
#
# The diversification comparison below does not compare "three projects"
# against "four projects". Instead, it compares two alternative ways of
# allocating the same incremental PPP exposure:
#   1. another transport-style PPP, which is highly exposed to the same
#      demand and sponsor-distress channels as the existing portfolio; and
#   2. a social-housing PPP, which carries meaningful routine fiscal exposure
#      but is less synchronised with transport-demand stress.
#
# This gives an economically coherent diversification result:
# the diversified case can have a higher mean/P50, while still reducing
# P95/P99 downside exposure.
baseline_availability_payment_d = {
    year: 70 * (1.025 ** (year - 2030))
    for year in YEARS
}

rows = []

# Store portfolio-level comparison rows separately.
# This allows the diversification analysis to compare two same-sized
# incremental investment choices:
#   1. Concentrated portfolio: add another transport-style availability PPP.
#   2. Diversified portfolio: add a low-correlation social-housing PPP.
comparison_rows = []

for sim in range(N_SIMS):

    cumulative_revenue_factor = 1.0
    sponsor_failed = False

    for year in YEARS:

        if year in SHOCK_YEARS:
            z = np.random.normal(size=3) @ L.T

            annual_demand_shock = max(0, BASE_DEMAND_SHOCK + 0.02 * z[0])
            cost_shock = max(0, BASE_COST_SHOCK + 0.015 * z[1])
            financing_shock = max(0, BASE_FINANCING_SHOCK + 0.0075 * z[2])

            cumulative_revenue_factor *= (1 - annual_demand_shock)

        else:
            # Shock ends, but revenue remains on the lower path.
            # Growth resumes because the baseline itself continues growing.
            annual_demand_shock = 0.0
            cost_shock = 0.0
            financing_shock = 0.0

        # -----------------------------
        # Project A: pure concession
        # -----------------------------

        project_a_loss = 0.0

        # -----------------------------
        # Project B: 80% revenue guarantee
        # -----------------------------

        base_revenue = baseline_revenue_b[year]
        guarantee_floor = 0.80 * base_revenue
        stressed_revenue = base_revenue * cumulative_revenue_factor

        project_b_loss = max(0, guarantee_floor - stressed_revenue)

        # --------------------------------------------------
        # Project C: Eastern Ring Road PPP
        # Transport availability PPP
        # --------------------------------------------------

        base_payment_c = baseline_availability_payment_c[year]

        project_c_loss = base_payment_c * (
            1.00 * cost_shock +
            0.40 * financing_shock
        )

        # --------------------------------------------------
        # Project D: National Social Housing PPP
        # Diversifying availability PPP
        # --------------------------------------------------

        base_payment_d = baseline_availability_payment_d[year]

        project_d_loss = base_payment_d * (
            # Routine availability/lifecycle variation.
            # This raises the mean and P50, so the diversified case is not
            # mechanically "cheaper"; its benefit should appear in the tail.
            0.020 +
            0.85 * cost_shock +
            0.35 * financing_shock
        )

        # Counterfactual benchmark:
        # What if the same incremental exposure as Project D were instead
        # allocated to another transport-style availability PPP?
        concentrated_incremental_loss = base_payment_d * (
            # Counterfactual: the same incremental PPP exposure is allocated
            # to another transport-style project. It therefore loads on the
            # same demand, cost, financing, and sponsor channels as the
            # existing transport portfolio.
            1.00 * cost_shock +
            0.40 * financing_shock +
            0.35 * annual_demand_shock
        )

        # --------------------------------------------------
        # Shared sponsor distress
        # One-off restructuring / intervention event
        # --------------------------------------------------

        severe_revenue_stress = cumulative_revenue_factor < 0.80

        during_shock_period = year in SHOCK_YEARS

        post_shock_stress_period = (
            year > SHOCK_YEARS[-1] and
            year <= SHOCK_YEARS[-1] + 3
        )

        shared_sponsor_failure = False

        if severe_revenue_stress and not sponsor_failed:

            if during_shock_period:
                shared_sponsor_failure = np.random.rand() < 0.25

            elif post_shock_stress_period:
                shared_sponsor_failure = np.random.rand() < 0.10

            if shared_sponsor_failure:
                sponsor_failed = True

                project_a_loss += 40.0
                project_b_loss += 20.0

        if shared_sponsor_failure:
            concentrated_incremental_loss += 40.0

        base_portfolio_loss = project_a_loss + project_b_loss + project_c_loss
        concentrated_portfolio_loss = (
            base_portfolio_loss + concentrated_incremental_loss
        )
        diversified_portfolio_loss = base_portfolio_loss + project_d_loss

        # --------------------------------------------------
        # Store project-level results
        # --------------------------------------------------

        rows.extend([
            {
                "simulation": sim,
                "year": year,
                "project": "Western Motorway PPP",
                "additional_government_payment_eur_m": project_a_loss,
                "shared_sponsor_failure": shared_sponsor_failure
            },
            {
                "simulation": sim,
                "year": year,
                "project": "Southern Tunnel PPP",
                "additional_government_payment_eur_m": project_b_loss,
                "shared_sponsor_failure": shared_sponsor_failure
            },
            {
                "simulation": sim,
                "year": year,
                "project": "Eastern Ring Road PPP",
                "additional_government_payment_eur_m": project_c_loss,
                "shared_sponsor_failure": False
            },
            {
                "simulation": sim,
                "year": year,
                "project": "National Social Housing PPP",
                "additional_government_payment_eur_m": project_d_loss,
                "shared_sponsor_failure": False
            }
        ])

        comparison_rows.extend([
            {
                "simulation": sim,
                "year": year,
                "portfolio": "Concentrated Transport Portfolio",
                "additional_government_payment_eur_m": concentrated_portfolio_loss,
            },
            {
                "simulation": sim,
                "year": year,
                "portfolio": "Diversified Portfolio",
                "additional_government_payment_eur_m": diversified_portfolio_loss,
            }
        ])


df = pd.DataFrame(rows)
df.to_csv("correlated_risk_simulation_full.csv", index=False)

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv("diversification_simulation_full.csv", index=False)

# --------------------------------------------------
# Select representative simulation paths
# using total portfolio loss over the full contract
# --------------------------------------------------

# Use only the original three-project transport portfolio for direct
# comparability with the earlier correlated-risk case-study section.
base_projects = [
    "Western Motorway PPP",
    "Southern Tunnel PPP",
    "Eastern Ring Road PPP",
]

portfolio_total = (
    df[df["project"].isin(base_projects)]
    .groupby("simulation")["additional_government_payment_eur_m"]
    .sum()
)

target_p50 = portfolio_total.quantile(0.50)
target_p95 = portfolio_total.quantile(0.95)
target_p99 = portfolio_total.quantile(0.99)

# Representative percentile simulations
rep_sims = {
    "Correlated Risk P50":
        (portfolio_total - target_p50).abs().idxmin(),

    "Correlated Risk P95":
        (portfolio_total - target_p95).abs().idxmin(),

    "Correlated Risk P99":
        (portfolio_total - target_p99).abs().idxmin(),
}

# --------------------------------------------------
# Add explicit sponsor distress scenario
# --------------------------------------------------

failure_sims = df.loc[
    df["shared_sponsor_failure"] == True,
    "simulation"
].unique()

if len(failure_sims) > 0:

    sponsor_distress_sim = portfolio_total.loc[failure_sims].idxmax()

    rep_sims["Sponsor Distress Scenario"] = sponsor_distress_sim

# --------------------------------------------------
# Build export table
# --------------------------------------------------

pfram_paths = []

for scenario, sim_id in rep_sims.items():

    sim_path = df[
        (df["simulation"] == sim_id) &
        (df["project"].isin(base_projects))
    ].copy()

    sim_path["scenario"] = scenario

    pfram_paths.append(sim_path)

pfram_paths = pd.concat(pfram_paths, ignore_index=True)

# --------------------------------------------------
# Pivot into PFRAM-ready format
# --------------------------------------------------

pfram_export = pfram_paths.pivot_table(
    index=["scenario", "project"],
    columns="year",
    values="additional_government_payment_eur_m"
).reset_index()

pfram_export.columns.name = None

for year in YEARS:
    pfram_export[year] = pfram_export[year].round(2)

# --------------------------------------------------
# Export
# --------------------------------------------------

pfram_export.to_csv(
    "pfram_project_payment_paths.csv",
    index=False
)

print("\nPFRAM PAYMENT PATHS\n")
print(pfram_export)


# --------------------------------------------------
# Publication-quality line plot of project payment paths
# --------------------------------------------------

plot_df = pfram_paths.copy()

# Aggregate to portfolio level
portfolio_plot = (
    plot_df.groupby(["scenario", "year"])["additional_government_payment_eur_m"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(10, 6))

for scenario in portfolio_plot["scenario"].unique():
    scenario_data = portfolio_plot[
        portfolio_plot["scenario"] == scenario
    ]

    plt.plot(
        scenario_data["year"],
        scenario_data["additional_government_payment_eur_m"],
        linewidth=2,
        marker="o",
        markersize=4,
        label=scenario
    )

plt.axvspan(
    min(SHOCK_YEARS),
    max(SHOCK_YEARS) + 1,
    alpha=0.12,
    label="Shock period"
)

plt.title(
    "Correlated Risk Overlay: Portfolio Fiscal Payment Paths",
    fontsize=14,
    weight="bold"
)

plt.xlabel("Year", fontsize=11)
plt.ylabel("Additional government payment (€m)", fontsize=11)

plt.xticks(YEARS, rotation=45)
plt.grid(True, linewidth=0.5, alpha=0.4)
plt.legend(frameon=False, fontsize=9)

plt.tight_layout()

plt.savefig("correlated_risk_portfolio_paths.png", dpi=300)
plt.savefig("correlated_risk_portfolio_paths.svg")

plt.close()


# --------------------------------------------------
# Publication-quality spaghetti plot of simulated portfolio paths
# --------------------------------------------------

N_PLOT_SIMS = 1000

# Aggregate project losses to base portfolio-year level
all_portfolio_paths = (
    df[df["project"].isin(base_projects)]
    .groupby(["simulation", "year"])["additional_government_payment_eur_m"]
    .sum()
    .reset_index()
)

# Select 1,000 simulations for visual clarity
plot_sims = np.random.choice(
    all_portfolio_paths["simulation"].unique(),
    size=N_PLOT_SIMS,
    replace=False
)

spaghetti_df = all_portfolio_paths[
    all_portfolio_paths["simulation"].isin(plot_sims)
]

plt.figure(figsize=(10, 6))

for sim_id, group in spaghetti_df.groupby("simulation"):
    plt.plot(
        group["year"],
        group["additional_government_payment_eur_m"],
        linewidth=0.5,
        alpha=0.08
    )

plt.axvspan(
    min(SHOCK_YEARS),
    max(SHOCK_YEARS) + 1,
    alpha=0.12,
    label="Shock period"
)

plt.title(
    "Simulated Portfolio Fiscal Payment Paths",
    fontsize=14,
    weight="bold"
)

plt.xlabel("Year", fontsize=11)
plt.ylabel("Additional government payment (€m)", fontsize=11)

plt.xticks(YEARS, rotation=45)
plt.grid(True, linewidth=0.5, alpha=0.4)

plt.tight_layout()

plt.savefig("simulated_portfolio_paths_spaghetti.png", dpi=300)
plt.savefig("simulated_portfolio_paths_spaghetti.svg")

plt.close()


# --------------------------------------------------
# Diversification analysis
# --------------------------------------------------

portfolio_variant_totals = (
    comparison_df
    .groupby(["portfolio", "simulation"])["additional_government_payment_eur_m"]
    .sum()
    .reset_index()
)

portfolio_variant_wide = portfolio_variant_totals.pivot(
    index="simulation",
    columns="portfolio",
    values="additional_government_payment_eur_m"
)

metrics = []

for portfolio_name in portfolio_variant_wide.columns:
    values = portfolio_variant_wide[portfolio_name]

    metrics.append({
        "portfolio": portfolio_name,
        "mean_total_payment_eur_m": values.mean(),
        "standard_deviation_eur_m": values.std(),
        "p50_total_payment_eur_m": values.quantile(0.50),
        "p95_total_payment_eur_m": values.quantile(0.95),
        "p99_total_payment_eur_m": values.quantile(0.99),
        "max_total_payment_eur_m": values.max(),
    })

diversification_metrics = pd.DataFrame(metrics)

concentrated_p99 = diversification_metrics.loc[
    diversification_metrics["portfolio"] == "Concentrated Transport Portfolio",
    "p99_total_payment_eur_m"
].iloc[0]

diversified_p99 = diversification_metrics.loc[
    diversification_metrics["portfolio"] == "Diversified Portfolio",
    "p99_total_payment_eur_m"
].iloc[0]

tail_reduction = (
    (concentrated_p99 - diversified_p99) / concentrated_p99
)

diversification_metrics["p99_tail_reduction_vs_concentrated"] = np.where(
    diversification_metrics["portfolio"] == "Diversified Portfolio",
    tail_reduction,
    np.nan
)

diversification_metrics = diversification_metrics.round(4)
diversification_metrics.to_csv("diversification_tail_metrics.csv", index=False)

print("\nDIVERSIFICATION TAIL METRICS\n")
print(diversification_metrics)
print(
    "\nP99 tail-risk reduction from diversification: "
    f"{tail_reduction:.2%}"
)

# --------------------------------------------------
# Distribution plot: concentrated vs diversified
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for portfolio_name in [
    "Concentrated Transport Portfolio",
    "Diversified Portfolio"
]:
    values = portfolio_variant_wide[portfolio_name]
    plt.hist(
        values,
        bins=60,
        alpha=0.45,
        density=True,
        label=portfolio_name
    )

plt.title(
    "Portfolio Diversification: Distribution of Total Fiscal Payments",
    fontsize=14,
    weight="bold"
)

plt.xlabel("Total additional government payment over contract (€m)", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.grid(True, linewidth=0.5, alpha=0.4)
plt.legend(frameon=False, fontsize=9)

plt.tight_layout()

plt.savefig("diversification_loss_distribution.png", dpi=300)
plt.savefig("diversification_loss_distribution.svg")

plt.close()

# --------------------------------------------------
# Tail metrics bar chart
# --------------------------------------------------

tail_plot = diversification_metrics.set_index("portfolio")[
    ["p95_total_payment_eur_m", "p99_total_payment_eur_m"]
]

ax = tail_plot.plot(kind="bar", figsize=(10, 6))

plt.title(
    "Portfolio Diversification: Tail Fiscal Exposure",
    fontsize=14,
    weight="bold"
)

plt.xlabel("")
plt.ylabel("Total additional government payment (€m)", fontsize=11)
plt.xticks(rotation=0)
plt.grid(axis="y", linewidth=0.5, alpha=0.4)
plt.legend(
    ["P95", "P99"],
    frameon=False,
    fontsize=9
)

plt.tight_layout()

plt.savefig("diversification_tail_metrics.png", dpi=300)
plt.savefig("diversification_tail_metrics.svg")

plt.close()




# --------------------------------------------------
# Diversification comparison chart
# Refactored: percentile on x-axis,
# portfolio type as legend
# --------------------------------------------------

tail_plot_refactored = (
    diversification_metrics
    .set_index("portfolio")
    .loc[
        [
            "Concentrated Transport Portfolio",
            "Diversified Portfolio"
        ],
        [
            "p50_total_payment_eur_m",
            "p95_total_payment_eur_m",
            "p99_total_payment_eur_m"
        ]
    ]
    .rename(
        columns={
            "p50_total_payment_eur_m": "P50",
            "p95_total_payment_eur_m": "P95",
            "p99_total_payment_eur_m": "P99"
        },
        index={
            "Concentrated Transport Portfolio": "Concentrated Portfolio",
            "Diversified Portfolio": "Diversified Portfolio"
        }
    )
    .T
)

ax = tail_plot_refactored.plot(
    kind="bar",
    figsize=(8, 5),
    width=0.75
)

plt.xlabel("Portfolio Loss Percentile", fontsize=11)
plt.ylabel("Total Fiscal Exposure (€m)", fontsize=11)

plt.title(
    "Tail Fiscal Exposure: Concentrated vs Diversified Portfolio",
    fontsize=13,
    weight="bold"
)

plt.xticks(rotation=0)
plt.grid(axis="y", linewidth=0.4, alpha=0.3)
plt.legend(frameon=False, fontsize=9)

plt.tight_layout()

plt.savefig("tail_risk_comparison_refactored.png", dpi=300)
plt.savefig("tail_risk_comparison_refactored.svg")

plt.close()
