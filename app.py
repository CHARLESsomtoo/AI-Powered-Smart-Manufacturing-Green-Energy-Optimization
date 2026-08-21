import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Smart Manufacturing",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_project_data():

    maintenance = pd.read_csv(
        "project_outputs/maintenance_profile.csv"
    )

    energy = pd.read_csv(
        "project_outputs/energy_profile.csv"
    )

    models = pd.read_csv(
        "project_outputs/model_comparison.csv"
    )

    return maintenance, energy, models


@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="CS01/predictive-maintenance-model",
        filename="predictive_maintenance_model_small.pkl",
        repo_type="model"
    )

    model = joblib.load(model_path)

    features = joblib.load(
        "models/model_features.pkl"
    )

    return model, features


@st.cache_data
def load_raw_energy():

    files = [
        "data/Energy_dataset.csv",
        "data/Energy_dataset",
        "data/smart_manufacturing_data.csv",
        "data/smart_manufacturing_data"
    ]

    for file in files:

        if os.path.exists(file):
            return pd.read_csv(file)

    return None


# ============================================================
# LOAD
# ============================================================

maintenance, energy, model_comparison = load_project_data()

model, features = load_model()

raw_energy = load_raw_energy()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Executive Dashboard",
        "Predictive Maintenance",
        "Energy Optimization",
        "Recommendations",
        "About the Project"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "AI-Powered Smart Manufacturing & Green Energy Optimization"
)

st.caption(
    "AI-powered decision-support system combining predictive "
    "maintenance and industrial energy optimization."
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.header("Executive Dashboard")

    maintenance_rate = (
        maintenance["actual_maintenance_rate"].mean()
    )

    avg_risk = (
        maintenance["avg_maintenance_probability"].mean()
    )

    avg_efficiency = (
        energy["avg_efficiency"].mean()
    )

    priority_count = (
        energy["energy_performance"]
        .eq("Optimization Priority")
        .sum()
    )

    machine_count = (
        maintenance["machine_id"].nunique()
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Maintenance Rate",
        f"{maintenance_rate:.1f}%"
    )

    c2.metric(
        "Average Maintenance Risk",
        f"{avg_risk:.1%}"
    )

    c3.metric(
        "Average Energy Efficiency",
        f"{avg_efficiency:.2f}"
    )

    c4.metric(
        "Optimization Priorities",
        int(priority_count)
    )

    c5.metric(
        "Manufacturing Machines",
        machine_count
    )

    st.divider()

    st.subheader(
        "Top 10 Machines by Maintenance Probability"
    )

    top_risk = (
        maintenance[
            [
                "machine_id",
                "avg_maintenance_probability"
            ]
        ]
        .sort_values(
            "avg_maintenance_probability",
            ascending=False
        )
        .head(10)
        .copy()
    )

    top_risk["Risk (%)"] = (
        top_risk["avg_maintenance_probability"] * 100
    )

    st.bar_chart(
        top_risk.set_index("machine_id")["Risk (%)"]
    )

    st.subheader(
        "Energy Efficiency by Machine"
    )

    efficiency = (
        energy
        .sort_values(
            "avg_efficiency",
            ascending=False
        )
        .set_index("machine_id")["avg_efficiency"]
    )

    st.bar_chart(efficiency)

    st.subheader("Key Findings")

    best_machine = energy.loc[
        energy["avg_efficiency"].idxmax(),
        "machine_id"
    ]

    priority_rows = energy[
        energy["energy_performance"]
        == "Optimization Priority"
    ]

    if not priority_rows.empty:

        priority_machine = priority_rows.iloc[0]["machine_id"]

    else:

        priority_machine = "None"

    st.success(
        f"Most energy-efficient machine: **{best_machine}**"
    )

    st.warning(
        f"Optimization priority machine: **{priority_machine}**"
    )


# ============================================================
# PREDICTIVE MAINTENANCE
# ============================================================

if page == "Predictive Maintenance":

    st.header("🤖 Predictive Maintenance")

    st.write(
        "The Random Forest model estimates maintenance probability "
        "from machine operating conditions."
    )

    machines = sorted(
        maintenance["machine_id"]
        .dropna()
        .astype(int)
        .unique()
    )

    selected_machine = st.selectbox(
        "Select Machine",
        machines
    )

    selected = maintenance[
        maintenance["machine_id"]
        == selected_machine
    ].iloc[0]

    st.subheader(
        f"Machine {selected_machine} Risk Profile"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Maintenance Probability",
        f"{selected['avg_maintenance_probability']:.1%}"
    )

    c2.metric(
        "Actual Maintenance Rate",
        f"{selected['actual_maintenance_rate']:.1f}%"
    )

    c3.metric(
        "Predicted Maintenance Rate",
        f"{selected['predicted_maintenance_rate']:.1f}%"
    )

    st.subheader(
        "Operating Conditions"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Temperature",
        f"{selected['avg_temperature']:.2f}"
    )

    c2.metric(
        "Vibration",
        f"{selected['avg_vibration']:.2f}"
    )

    c3.metric(
        "Energy Consumption",
        f"{selected['avg_energy_consumption']:.2f}"
    )

    st.subheader(
        "AI Feature Importance"
    )

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance["Importance (%)"] = (
        importance["Importance"] * 100
    )

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Feature")[
            "Importance (%)"
        ]
    )

    st.dataframe(
        importance[
            [
                "Feature",
                "Importance (%)"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Model Performance"
    )

    st.dataframe(
        model_comparison.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        Random Forest achieved the strongest overall performance.

        Accuracy: 89.1%
        Precision: 99.9%
        Recall: 44.9%
        F1 Score: 62.0%
        ROC-AUC: 72.4%

        Because recall is relatively low, the model should be used as a
        decision-support tool rather than an autonomous maintenance system.
        """
    )

    st.subheader(
        "Maintenance Probability Ranking"
    )

    ranking = maintenance.copy()

    ranking["Predicted Probability (%)"] = (
        ranking["avg_maintenance_probability"] * 100
    )

    st.dataframe(
        ranking[
            [
                "machine_id",
                "Predicted Probability (%)",
                "actual_maintenance_rate",
                "predicted_maintenance_rate"
            ]
        ]
        .sort_values(
            "Predicted Probability (%)",
            ascending=False
        )
        .round(2),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ENERGY OPTIMIZATION
# ============================================================

if page == "Energy Optimization":

    st.header("⚡ Green Energy Optimization")

    total_energy = energy["total_energy_kWh"].sum()

    avg_efficiency = energy["avg_efficiency"].mean()

    highest_efficiency = energy["avg_efficiency"].max()

    priority_count = (
        energy["energy_performance"]
        .eq("Optimization Priority")
        .sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Energy",
        f"{total_energy:,.1f} kWh"
    )

    c2.metric(
        "Average Efficiency",
        f"{avg_efficiency:.2f}"
    )

    c3.metric(
        "Highest Efficiency",
        f"{highest_efficiency:.2f}"
    )

    c4.metric(
        "Optimization Priorities",
        int(priority_count)
    )

    st.divider()

    st.subheader(
        "Energy Efficiency Ranking"
    )

    efficiency_chart = (
        energy
        .sort_values(
            "avg_efficiency",
            ascending=False
        )
        .set_index("machine_id")["avg_efficiency"]
    )

    st.bar_chart(efficiency_chart)

    st.subheader(
        "Energy Variance"
    )

    variance_chart = (
        energy
        .set_index("machine_id")["avg_energy_variance"]
    )

    st.bar_chart(variance_chart)

    st.caption(
        "Positive variance indicates energy consumption above forecast."
    )

    st.subheader(
        "Machine Energy Performance"
    )

    columns = [
        "machine_id",
        "avg_energy_kWh",
        "total_energy_kWh",
        "avg_efficiency",
        "avg_energy_variance",
        "energy_performance",
        "optimization_recommendation"
    ]

    st.dataframe(
        energy[columns]
        .sort_values(
            "avg_efficiency",
            ascending=False
        )
        .round(3),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CORRECT OPTIMIZATION PRIORITY
    # --------------------------------------------------------

    st.subheader(
        "Optimization Priority"
    )

    priority = energy[
        energy["energy_performance"]
        == "Optimization Priority"
    ]

    if not priority.empty:

        machine = priority.iloc[0]

        st.warning(
            f"""
            **Primary optimization candidate: {machine['machine_id']}**

            Energy efficiency:
            {machine['avg_efficiency']:.3f} output/kWh

            Average energy variance:
            {machine['avg_energy_variance']:.3f} kWh

            Classification:
            {machine['energy_performance']}

            Recommendation:
            {machine['optimization_recommendation']}
            """
        )

    else:

        st.success(
            "No machine is currently classified as an optimization priority."
        )

    # --------------------------------------------------------
    # RAW ENERGY ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "Production Mode Analysis"
    )

    if raw_energy is None:

        st.info(
            "Raw energy dataset was not found. "
            "Machine-level energy analysis remains available."
        )

    else:

        required = [
            "production_mode",
            "energy_kWh",
            "production_output_units"
        ]

        if all(
            column in raw_energy.columns
            for column in required
        ):

            raw = raw_energy.copy()

            raw["energy_efficiency"] = (
                raw["production_output_units"]
                /
                raw["energy_kWh"].replace(0, pd.NA)
            )

            mode = (
                raw.groupby("production_mode")
                .agg(
                    avg_energy=(
                        "energy_kWh",
                        "mean"
                    ),
                    avg_production=(
                        "production_output_units",
                        "mean"
                    ),
                    avg_efficiency=(
                        "energy_efficiency",
                        "mean"
                    ),
                    observations=(
                        "production_mode",
                        "size"
                    )
                )
                .round(3)
            )

            st.dataframe(
                mode,
                use_container_width=True
            )

            st.bar_chart(
                mode["avg_efficiency"]
            )

        else:

            st.info(
                "The raw energy dataset does not contain the "
                "required production-mode columns."
            )

    # --------------------------------------------------------
    # UTILIZATION
    # --------------------------------------------------------

    if raw_energy is not None:

        required_utilization = [
            "machine_id",
            "machine_utilization_%",
            "energy_kWh",
            "production_output_units"
        ]

        if all(
            column in raw_energy.columns
            for column in required_utilization
        ):

            st.subheader(
                "Machine Utilization Analysis"
            )

            utilization = (
                raw_energy
                .groupby("machine_id")
                .agg(
                    avg_utilization=(
                        "machine_utilization_%",
                        "mean"
                    ),
                    avg_energy=(
                        "energy_kWh",
                        "mean"
                    ),
                    avg_production=(
                        "production_output_units",
                        "mean"
                    )
                )
                .round(3)
            )

            st.dataframe(
                utilization,
                use_container_width=True
            )

            st.subheader(
                "Utilization Correlation"
            )

            correlation = raw_energy[
                [
                    "machine_utilization_%",
                    "energy_kWh",
                    "production_output_units"
                ]
            ].corr()

            st.dataframe(
                correlation.round(3),
                use_container_width=True
            )


# ============================================================
# RECOMMENDATIONS
# ============================================================

if page == "Recommendations":

    st.header(
        "💡 AI Decision-Support Recommendations"
    )

    st.subheader(
        "Energy Optimization"
    )

    for _, row in energy.iterrows():

        machine = row["machine_id"]

        classification = row["energy_performance"]

        efficiency = row["avg_efficiency"]

        variance = row["avg_energy_variance"]

        recommendation = row[
            "optimization_recommendation"
        ]

        if classification == "Optimization Priority":

            st.error(
                f"""
                **{machine} — OPTIMIZATION PRIORITY**

                Efficiency: {efficiency:.3f} output/kWh

                Energy variance: {variance:.3f} kWh

                {recommendation}
                """
            )

        elif classification == "Less Efficient but Below Forecast":

            st.warning(
                f"""
                **{machine} — EFFICIENCY IMPROVEMENT**

                Efficiency: {efficiency:.3f} output/kWh

                Energy variance: {variance:.3f} kWh

                {recommendation}
                """
            )

        elif classification == "Efficient but Above Forecast":

            st.warning(
                f"""
                **{machine} — MONITOR ENERGY**

                Efficiency: {efficiency:.3f} output/kWh

                Energy variance: {variance:.3f} kWh

                {recommendation}
                """
            )

        else:

            st.success(
                f"""
                **{machine} — EFFICIENT**

                Efficiency: {efficiency:.3f} output/kWh

                Energy variance: {variance:.3f} kWh

                {recommendation}
                """
            )

    st.subheader(
        "Predictive Maintenance"
    )

    top_risk = (
        maintenance
        .sort_values(
            "avg_maintenance_probability",
            ascending=False
        )
        .head(10)
    )

    for _, row in top_risk.iterrows():

        st.info(
            f"""
            **Machine {int(row['machine_id'])}**

            Predicted maintenance probability:
            {row['avg_maintenance_probability']:.1%}

            Recommendation:
            Continue monitoring temperature and vibration and
            prioritize inspection where operational conditions deteriorate.
            """
        )

    st.divider()

    st.success(
        """
        **Overall strategy**

        Use predictive maintenance to prioritize machine inspections
        and energy analytics to identify efficiency opportunities.

        Recommendations should be validated by engineering and
        operations teams before implementation.
        """
    )


# ============================================================
# ABOUT
# ============================================================

if page == "About the Project":

    st.header(
        "About the Project"
    )

    st.markdown(
        """
        ## AI-Powered Smart Manufacturing & Green Energy Optimization

        This project combines machine learning and industrial energy
        analytics to create a decision-support system for manufacturing.

        ### Predictive Maintenance

        The Random Forest model uses:

        - Temperature
        - Vibration
        - Humidity
        - Pressure
        - Energy consumption

        to estimate maintenance requirements.

        ### Energy Optimization

        The project evaluates:

        - Energy consumption
        - Production output
        - Energy efficiency
        - Energy variance
        - Machine utilization
        - Production mode

        ### Key Findings

        **MCH_3** has the highest energy efficiency.

        **MCH_2** has the lowest energy efficiency.

        **MCH_5** has the largest positive energy variance.

        **MCH_4** is the identified optimization-priority machine.

        ### Limitation

        The predictive model is a decision-support tool and should not
        replace engineering judgement.
        """
    )
