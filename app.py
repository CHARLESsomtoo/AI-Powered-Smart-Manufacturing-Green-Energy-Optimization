import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI-Powered Smart Manufacturing & Green Energy Optimization",
    page_icon="🏭",
    layout="wide"
)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "predictive_maintenance_model_small.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"
DATA_DIR = BASE_DIR / "data"


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def load_features():
    if not FEATURES_PATH.exists():
        return []

    try:
        features = joblib.load(FEATURES_PATH)

        if isinstance(features, (list, tuple, np.ndarray, pd.Index)):
            return list(features)

        return []

    except Exception:
        return []


def load_dataset(possible_names):
    for name in possible_names:
        file_path = DATA_DIR / name

        if file_path.exists():
            try:
                if file_path.suffix.lower() == ".csv":
                    return pd.read_csv(file_path)

                if file_path.suffix.lower() in [".xlsx", ".xls"]:
                    return pd.read_excel(file_path)

            except Exception:
                pass

    return pd.DataFrame()


def safe_column(df, column_name, default=0):
    if column_name in df.columns:
        return df[column_name]
    return pd.Series([default] * len(df), index=df.index)


# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

model = load_model()
model_features = load_features()

manufacturing_df = load_dataset(
    [
        "smart_manufacturing_data.csv",
        "smart_manufacturing_data.xlsx",
        "smart_manufacturing_data.xls"
    ]
)

energy_df = load_dataset(
    [
        "Energy_dataset.csv",
        "Energy_dataset.xlsx",
        "Energy_dataset.xls",
        "energy_dataset.csv",
        "energy_dataset.xlsx",
        "energy_dataset.xls"
    ]
)


# --------------------------------------------------
# APP TITLE
# --------------------------------------------------

st.title("AI-Powered Smart Manufacturing & Green Energy Optimization")

st.caption(
    "An AI-powered decision-support framework combining predictive "
    "maintenance and industrial energy optimization."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Executive Dashboard",
        "Predictive Maintenance",
        "Energy Optimization",
        "Recommendations"
    ]
)


# --------------------------------------------------
# EXECUTIVE DASHBOARD
# --------------------------------------------------

if page == "Executive Dashboard":

    st.header("Executive Dashboard")

    if manufacturing_df.empty:
        st.error("Smart manufacturing dataset could not be found.")
    else:

        numeric_df = manufacturing_df.select_dtypes(
            include=np.number
        )

        machine_count = len(manufacturing_df)

        maintenance_rate = 0

        maintenance_columns = [
            "maintenance_probability",
            "avg_maintenance_probability",
            "actual_maintenance_rate",
            "predicted_maintenance_rate"
        ]

        for column in maintenance_columns:
            if column in manufacturing_df.columns:
                maintenance_rate = (
                    manufacturing_df[column].mean()
                )
                break

        if maintenance_rate <= 1:
            maintenance_rate = maintenance_rate * 100

        energy_efficiency = 0

        energy_columns = [
            "energy_efficiency",
            "Energy_Efficiency",
            "avg_energy_efficiency"
        ]

        for column in energy_columns:
            if column in manufacturing_df.columns:
                energy_efficiency = (
                    manufacturing_df[column].mean()
                )
                break

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Maintenance Rate",
            f"{maintenance_rate:.1f}%"
        )

        col2.metric(
            "Average Energy Efficiency",
            f"{energy_efficiency:.2f}"
        )

        col3.metric(
            "Optimization Priorities",
            "1"
        )

        col4.metric(
            "Manufacturing Machines",
            machine_count
        )

        st.subheader("Maintenance Risk by Machine")

        risk_column = None

        possible_risk_columns = [
            "avg_maintenance_probability",
            "maintenance_probability",
            "predicted_maintenance_rate",
            "actual_maintenance_rate"
        ]

        for column in possible_risk_columns:
            if column in manufacturing_df.columns:
                risk_column = column
                break

        if risk_column is not None:

            chart_df = pd.DataFrame(
                {
                    "Machine": np.arange(
                        1,
                        len(manufacturing_df) + 1
                    ),
                    "Maintenance Risk": manufacturing_df[
                        risk_column
                    ].values
                }
            )

            st.bar_chart(
                chart_df.set_index("Machine")
            )

        else:
            st.info(
                "Maintenance risk column was not found in the dataset."
            )

        st.subheader("Energy Efficiency by Machine")

        efficiency_column = None

        possible_efficiency_columns = [
            "energy_efficiency",
            "Energy_Efficiency",
            "avg_energy_efficiency"
        ]

        for column in possible_efficiency_columns:
            if column in manufacturing_df.columns:
                efficiency_column = column
                break

        if efficiency_column is not None:

            efficiency_chart = pd.DataFrame(
                {
                    "Machine": np.arange(
                        1,
                        len(manufacturing_df) + 1
                    ),
                    "Energy Efficiency": manufacturing_df[
                        efficiency_column
                    ].values
                }
            )

            st.bar_chart(
                efficiency_chart.set_index("Machine")
            )

        else:

            if len(numeric_df.columns) > 0:
                sample_efficiency = numeric_df.mean().head(5)

                st.bar_chart(sample_efficiency)

            else:
                st.info(
                    "Energy efficiency data could not be displayed."
                )


# --------------------------------------------------
# PREDICTIVE MAINTENANCE
# --------------------------------------------------

elif page == "Predictive Maintenance":

    st.header("Predictive Maintenance")

    if manufacturing_df.empty:
        st.error(
            "Smart manufacturing dataset could not be found."
        )

    else:

        if model is None:
            st.error(
                "Model file not found. Expected file: "
                "predictive_maintenance_model_small.pkl"
            )

        else:

            st.success(
                "Predictive maintenance model loaded successfully."
            )

            st.subheader("Machine Maintenance Risk")

            if model_features:

                prediction_input = pd.DataFrame(
                    index=manufacturing_df.index
                )

                for feature in model_features:

                    if feature in manufacturing_df.columns:

                        prediction_input[feature] = pd.to_numeric(
                            manufacturing_df[feature],
                            errors="coerce"
                        ).fillna(0)

                    else:

                        prediction_input[feature] = 0

                try:

                    probabilities = model.predict_proba(
                        prediction_input
                    )[:, 1]

                    maintenance_results = manufacturing_df.copy()

                    maintenance_results[
                        "maintenance_probability"
                    ] = probabilities

                    maintenance_results[
                        "maintenance_risk"
                    ] = np.where(
                        probabilities >= 0.70,
                        "High Risk",
                        np.where(
                            probabilities >= 0.40,
                            "Medium Risk",
                            "Low Risk"
                        )
                    )

                    display_columns = []

                    if "machine_id" in maintenance_results.columns:
                        display_columns.append("machine_id")

                    for column in [
                        "avg_temperature",
                        "avg_vibration",
                        "avg_energy_consumption"
                    ]:
                        if column in maintenance_results.columns:
                            display_columns.append(column)

                    display_columns.extend(
                        [
                            "maintenance_probability",
                            "maintenance_risk"
                        ]
                    )

                    display_columns = list(
                        dict.fromkeys(display_columns)
                    )

                    st.dataframe(
                        maintenance_results[
                            display_columns
                        ].sort_values(
                            "maintenance_probability",
                            ascending=False
                        ),
                        use_container_width=True
                    )

                    st.subheader(
                        "Maintenance Risk Distribution"
                    )

                    risk_counts = (
                        maintenance_results[
                            "maintenance_risk"
                        ]
                        .value_counts()
                    )

                    st.bar_chart(risk_counts)

                except Exception as error:

                    st.error(
                        f"Model prediction error: {error}"
                    )

            else:

                st.error(
                    "model_features.pkl could not be loaded."
                )


# --------------------------------------------------
# ENERGY OPTIMIZATION
# --------------------------------------------------

elif page == "Energy Optimization":

    st.header("Energy Optimization")

    if energy_df.empty and manufacturing_df.empty:

        st.error(
            "Energy dataset could not be found."
        )

    else:

        source_df = energy_df.copy()

        if source_df.empty:
            source_df = manufacturing_df.copy()

        st.subheader("Optimization Priority")

        energy_column = None

        possible_energy_columns = [
            "energy_consumption",
            "Energy_Consumption",
            "avg_energy_consumption",
            "energy_efficiency",
            "Energy_Efficiency"
        ]

        for column in possible_energy_columns:
            if column in source_df.columns:
                energy_column = column
                break

        if energy_column is not None:

            source_df[energy_column] = pd.to_numeric(
                source_df[energy_column],
                errors="coerce"
            )

            priority_row = source_df.loc[
                source_df[energy_column].idxmax()
            ]

            if "machine_id" in source_df.columns:
                priority_machine = priority_row["machine_id"]
            else:
                priority_machine = priority_row.name

            priority_value = priority_row[energy_column]

            st.warning(
                f"Primary optimization candidate: "
                f"{priority_machine}"
            )

            st.write(
                f"Energy indicator value: "
                f"{priority_value:.2f}"
            )

            st.subheader("Machine Energy Analysis")

            if "machine_id" in source_df.columns:

                energy_chart = source_df[
                    ["machine_id", energy_column]
                ].copy()

                energy_chart = energy_chart.set_index(
                    "machine_id"
                )

                st.bar_chart(energy_chart)

            else:

                st.bar_chart(
                    source_df[energy_column]
                )

        else:

            st.info(
                "No recognized energy consumption or energy "
                "efficiency column was found."
            )

        if not energy_df.empty:

            csv = energy_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Energy Optimization Data",
                data=csv,
                file_name="energy_optimization_data.csv",
                mime="text/csv"
            )


# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------

elif page == "Recommendations":

    st.header("Business Recommendations")

    st.subheader("Predictive Maintenance")

    st.write(
        "Prioritize machines with high predicted maintenance "
        "probability for preventive inspection and maintenance."
    )

    st.subheader("Energy Efficiency")

    st.write(
        "Prioritize machines with high energy consumption or "
        "low energy efficiency for operational optimization."
    )

    st.subheader("Operational Strategy")

    st.write(
        "Combine maintenance risk predictions with energy "
        "performance to support production planning and "
        "resource allocation."
    )

    st.subheader("Management Recommendation")

    st.write(
        "Use the dashboard as a decision-support system to "
        "identify maintenance priorities, reduce unplanned "
        "downtime, and improve industrial energy efficiency."
    )
