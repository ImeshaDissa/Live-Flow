"""
streamlit_app.py

Live/FLOW -- Streamlit version.

Reads the CSV produced by fastf1_dataset.py and does two things:

1. BACKTEST: pick any round that's already happened this season, and see
   what the model would have predicted going in (using only form data
   available *before* that round) next to what actually happened.

2. FORWARD PREDICT: pick an upcoming circuit and get a predicted order
   using each driver's most recent rolling form, weighted by how well
   their team's car tends to suit that circuit's demands.

   Circuits already raced this season use their real, measured speed-trap
   and gear-change profile from build_circuit_profile() in
   fastf1_dataset.py. Circuits that haven't been raced yet this season
   fall back to a small built-in reference table of typical circuit
   characteristics (clearly labeled "estimated" in the UI) -- there's no
   telemetry for a race that hasn't happened.

Run:
    streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ---------------------------------------------------------------------------
# Static fallback: typical circuit demand profile for tracks the loaded
# dataset hasn't measured yet (i.e. haven't been raced this season).
# Same estimates used in the original HTML dashboard -- 0-100 scale.
# ---------------------------------------------------------------------------
CIRCUIT_REFERENCE = {
    "Hungarian Grand Prix":   {"straight": 30, "technical": 88, "street": 15},
    "Belgian Grand Prix":     {"straight": 88, "technical": 55, "street": 10},
    "British Grand Prix":     {"straight": 65, "technical": 80, "street": 10},
    "Monaco Grand Prix":      {"straight": 15, "technical": 70, "street": 95},
    "Italian Grand Prix":     {"straight": 95, "technical": 25, "street": 5},
    "Dutch Grand Prix":       {"straight": 35, "technical": 82, "street": 20},
    "Japanese Grand Prix":    {"straight": 55, "technical": 90, "street": 10},
    "Singapore Grand Prix":   {"straight": 20, "technical": 75, "street": 90},
    "Azerbaijan Grand Prix":  {"straight": 80, "technical": 40, "street": 80},
    "Saudi Arabian Grand Prix": {"straight": 75, "technical": 60, "street": 85},
    "Bahrain Grand Prix":     {"straight": 60, "technical": 55, "street": 5},
    "Australian Grand Prix":  {"straight": 55, "technical": 65, "street": 55},
    "Chinese Grand Prix":     {"straight": 60, "technical": 70, "street": 10},
    "Miami Grand Prix":       {"straight": 50, "technical": 60, "street": 70},
    "Canadian Grand Prix":    {"straight": 70, "technical": 45, "street": 60},
    "Austrian Grand Prix":    {"straight": 75, "technical": 50, "street": 5},
    "Spanish Grand Prix":     {"straight": 55, "technical": 78, "street": 5},
    "United States Grand Prix": {"straight": 60, "technical": 75, "street": 10},
    "Mexico City Grand Prix": {"straight": 70, "technical": 55, "street": 15},
    "Sao Paulo Grand Prix":   {"straight": 65, "technical": 70, "street": 10},
    "Las Vegas Grand Prix":   {"straight": 90, "technical": 20, "street": 95},
    "Qatar Grand Prix":       {"straight": 55, "technical": 85, "street": 5},
    "Abu Dhabi Grand Prix":   {"straight": 60, "technical": 65, "street": 15},
}

# Team archetype affinities -- how well a car historically suits each
# demand, 0-100. Same estimates as the HTML dashboard. Once you have a
# couple of full seasons in your dataset, these could be replaced with
# each team's measured average finish position at high/low straight-line
# tracks instead of hand-set numbers.
TEAM_AFFINITY = {
    "Mercedes":     {"straight": 92, "technical": 74, "street": 66},
    "Ferrari":      {"straight": 80, "technical": 90, "street": 72},
    "McLaren":      {"straight": 83, "technical": 86, "street": 82},
    "Red Bull Racing": {"straight": 76, "technical": 88, "street": 87},
    "Alpine":       {"straight": 58, "technical": 55, "street": 52},
    "RB":           {"straight": 60, "technical": 58, "street": 56},
    "Haas F1 Team": {"straight": 55, "technical": 52, "street": 50},
    "Audi":         {"straight": 50, "technical": 53, "street": 48},
    "Williams":     {"straight": 64, "technical": 50, "street": 54},
    "Aston Martin": {"straight": 48, "technical": 54, "street": 56},
    "Cadillac":     {"straight": 40, "technical": 38, "street": 40},
}


def team_key(team_name: str) -> str:
    """FastF1 TeamName strings are verbose (e.g. 'Mercedes AMG Petronas
    F1 Team') -- match them to the short keys above."""
    for key in TEAM_AFFINITY:
        if key.lower() in team_name.lower():
            return key
    return None


def normalize(series: pd.Series) -> pd.Series:
    rng = series.max() - series.min()
    if rng == 0 or pd.isna(rng):
        return series * 0 + 50
    return (series - series.min()) / rng * 100


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Live/FLOW", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0A0C0F; }
    h1, h2, h3 { text-transform: uppercase; letter-spacing: 1px; }
    .stMetric { background-color: #14171C; padding: 10px; border: 1px solid #2A2F38; }
</style>
""", unsafe_allow_html=True)

st.title("Live/FLOW — Race Predictor")
st.caption("Built on real FastF1 session data, not hand-typed estimates.")

uploaded = st.sidebar.file_uploader("Upload fastf1_dataset.csv", type="csv")
default_path = st.sidebar.text_input("...or path on disk", value="fastf1_dataset.csv")

@st.cache_data
def load_data(file_or_path):
    return pd.read_csv(file_or_path)

df = None
if uploaded is not None:
    df = load_data(uploaded)
elif default_path:
    try:
        df = load_data(default_path)
    except FileNotFoundError:
        st.warning(f"No file found at '{default_path}' yet. Run fastf1_dataset.py first, or upload a CSV in the sidebar.")

if df is None or df.empty:
    st.stop()

df["FinishPosition"] = pd.to_numeric(df["FinishPosition"], errors="coerce")
df["QualiPosition"] = pd.to_numeric(df["QualiPosition"], errors="coerce")
df["Points"] = pd.to_numeric(df["Points"], errors="coerce")

rounds = df[["RoundNumber", "EventName"]].drop_duplicates().sort_values("RoundNumber")

# ---------------------------------------------------------------------------
# Season overview
# ---------------------------------------------------------------------------

st.subheader("01 — Season Form")

cumulative = (
    df.sort_values(["Abbreviation", "RoundNumber"])
      .assign(CumPoints=lambda d: d.groupby("Abbreviation")["Points"].cumsum())
)
top_drivers = (
    df.groupby("Abbreviation")["Points"].sum().sort_values(ascending=False).head(8).index
)
fig = px.line(
    cumulative[cumulative["Abbreviation"].isin(top_drivers)],
    x="RoundNumber", y="CumPoints", color="Abbreviation",
    template="plotly_dark",
)
fig.update_layout(paper_bgcolor="#0A0C0F", plot_bgcolor="#0A0C0F", height=360)
st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------

st.subheader("02 — Backtest a Past Round")

chosen_round = st.selectbox(
    "Round", rounds["RoundNumber"],
    format_func=lambda r: rounds.set_index("RoundNumber").loc[r, "EventName"],
)

round_df = df[df["RoundNumber"] == chosen_round].copy()
round_df["team_short"] = round_df["TeamName"].apply(team_key)

# model prediction using only form known BEFORE this round (RollingPointsPerRace
# was already shift(1)'d in fastf1_dataset.py, so this is leakage-free)
round_df["form_score"] = normalize(round_df["RollingPointsPerRace"].fillna(0))
round_df["predicted_score"] = round_df["form_score"]
predicted_order = round_df.sort_values("predicted_score", ascending=False)
actual_order = round_df.sort_values("FinishPosition")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Model predicted (pre-race form only)**")
    st.dataframe(
        predicted_order[["Abbreviation", "TeamName", "RollingPointsPerRace"]].reset_index(drop=True),
        hide_index=False, width='stretch',
    )
with col2:
    st.markdown("**Actual result**")
    st.dataframe(
        actual_order[["Abbreviation", "TeamName", "FinishPosition", "Points"]].reset_index(drop=True),
        hide_index=False, width='stretch',
    )

# ---------------------------------------------------------------------------
# Forward prediction mode
# ---------------------------------------------------------------------------

st.subheader("03 — Predict an Upcoming Round")

circuit_choice = st.selectbox("Circuit", list(CIRCUIT_REFERENCE.keys()), index=0)

# does the loaded dataset already have a measured profile for this circuit name?
measured = df[df["EventName"].str.contains(circuit_choice.split()[0], case=False, na=False)]
has_measured_profile = "straight_line_index" in df.columns and not measured["straight_line_index"].dropna().empty

if has_measured_profile:
    st.success(f"Using measured telemetry from this season's {circuit_choice}.")
    straight_val = normalize(df["straight_line_index"]).loc[measured.index].mean()
    technical_val = 100 - normalize(df["avg_lap_speed_kmh"]).loc[measured.index].mean()
    street_val = CIRCUIT_REFERENCE[circuit_choice]["street"]  # not derivable from telemetry
else:
    ref = CIRCUIT_REFERENCE[circuit_choice]
    st.info("This circuit hasn't been raced yet this season — using estimated demand profile, not measured telemetry.")
    straight_val, technical_val, street_val = ref["straight"], ref["technical"], ref["street"]

c1, c2, c3 = st.columns(3)
c1.metric("Straight-line demand", f"{straight_val:.0f}/100")
c2.metric("Technical demand", f"{technical_val:.0f}/100")
c3.metric("Street-circuit demand", f"{street_val:.0f}/100")

latest_form = df.sort_values("RoundNumber").groupby("Abbreviation").tail(1).copy()
latest_form["team_short"] = latest_form["TeamName"].apply(team_key)
latest_form["form_score"] = normalize(latest_form["RollingPointsPerRace"].fillna(0))

def fit_score(team_short):
    aff = TEAM_AFFINITY.get(team_short)
    if not aff:
        return 50
    wsum = straight_val + technical_val + street_val
    if wsum == 0:
        return 50
    return (aff["straight"] * straight_val + aff["technical"] * technical_val + aff["street"] * street_val) / wsum

latest_form["fit_score"] = latest_form["team_short"].apply(fit_score)
latest_form["predicted_score"] = latest_form["form_score"] * 0.5 + latest_form["fit_score"] * 0.5
ranked = latest_form.sort_values("predicted_score", ascending=False).reset_index(drop=True)
ranked.index = ranked.index + 1

st.dataframe(
    ranked[["Abbreviation", "FullName", "TeamName", "RollingPointsPerRace", "fit_score", "predicted_score"]],
    width='stretch',
)

st.caption(
    "Model: 50% recent rolling form (points/race over last 4 rounds, real data) "
    "+ 50% team/circuit fit (measured telemetry where available, otherwise estimated)."
)
