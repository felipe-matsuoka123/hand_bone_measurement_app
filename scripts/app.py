import streamlit as st
import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def lms_z(x: float, L: float, M: float, S: float) -> float:
    if x <= 0 or M <= 0 or S <= 0:
        return np.nan
    if abs(L) < 1e-12:
        return np.log(x / M) / S
    return (((x / M) ** L) - 1.0) / (L * S)


def lms_for_age(df: pd.DataFrame, bone: str, sex: str, age: float):
    sub = df[(df["Bone"] == bone) & (df["Sex"] == sex)].sort_values("Age")

    if sub.empty:
        return np.nan, np.nan, np.nan

    ages = sub["Age"].to_numpy()
    Ls = sub["L"].to_numpy()
    Ms = sub["M"].to_numpy()
    Ss = sub["S"].to_numpy()

    # clamp se idade fora do intervalo
    if age <= ages[0]:
        return Ls[0], Ms[0], Ss[0]
    if age >= ages[-1]:
        return Ls[-1], Ms[-1], Ss[-1]

    return (
        float(np.interp(age, ages, Ls)),
        float(np.interp(age, ages, Ms)),
        float(np.interp(age, ages, Ss)),
    )


def z_and_percentile(df: pd.DataFrame, bone: str, sex: str, age: float, x: float):
    L, M, S = lms_for_age(df, bone, sex, age)
    z = lms_z(x, L, M, S)
    p = float(100 * norm.cdf(z)) if np.isfinite(z) else np.nan
    return float(z), p


def norm_ppf(p: float) -> float:
    # Approximate inverse CDF for standard normal (Acklam's method).
    if p <= 0.0 or p >= 1.0:
        return np.nan
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


if hasattr(st, "cache_data"):
    @st.cache_data
    def load_csv(path: str):
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = df.columns.str.strip()
        return df
else:
    @st.cache(allow_output_mutation=True)
    def load_csv(path: str):
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = df.columns.str.strip()
        return df

st.title("Perfil metacarpofalangeano de crianças e adolescentes")
st.caption("Cálculo do z-score e percentis por osso, sexo e idade com base em curvas LMS.")

df = load_csv("./data/BoneMeasurements.csv")

left_col, right_col = st.columns([1, 1.3])

with left_col:
    st.subheader("Adicione as medidas")
    with st.form("params_form", clear_on_submit=False):
        sex = st.selectbox("Sexo", sorted(df["Sex"].unique()))
        age = st.number_input(
            "Idade",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=0.1,
            help="Idade cronológica.",
        )

        bone = st.selectbox(
            "Osso",
            sorted(df["Bone"].unique())
        )

        x = st.number_input(
            "Medida do osso",
            min_value=0.0,
            value=0.0,
            step=0.01,
            help="Mesma unidade usada no estudo.",
        )

        submitted = st.form_submit_button("Calcular")

    if submitted:
        z, p = z_and_percentile(df, bone, sex, age, x)

        if np.isnan(z):
            st.error("Não foi possível calcular o z-score para esta combinação.")
        else:
            st.markdown(
                f"""
                <div style="padding: 12px 14px; border: 1px solid #E2E8F0; border-radius: 10px; background: #F8FAFC;">
                  <div style="font-weight: 600; margin-bottom: 6px;">Resultados</div>
                  <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                    <div style="min-width: 120px;">
                      <div style="font-size: 13px; color: #475569;">Z-score</div>
                      <div style="font-size: 28px; font-weight: 700; color: #0F172A;">{z:.2f}</div>
                    </div>
                    <div style="min-width: 120px;">
                      <div style="font-size: 13px; color: #475569;">Percentil</div>
                      <div style="font-size: 28px; font-weight: 700; color: #0F172A;">{p:.1f}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with right_col:
    st.subheader("Curva do percentil")
    if "z" in locals() and np.isfinite(z):
        age_grid = np.linspace(df["Age"].min(), df["Age"].max(), 200)

        sub = df[(df["Bone"] == bone) & (df["Sex"] == sex)].sort_values("Age")
        L = np.interp(age_grid, sub["Age"], sub["L"])
        M = np.interp(age_grid, sub["Age"], sub["M"])
        S = np.interp(age_grid, sub["Age"], sub["S"])


        fig, ax = plt.subplots(figsize=(10, 7))
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for percentile in percentiles:
            z_curve = norm.ppf(percentile / 100.0)
            if abs(z_curve) < 1e-12:
                y_curve = M
            elif np.any(np.abs(L) < 1e-12):
                y_curve = M * np.exp(S * z_curve)
            else:
                y_curve = M * (1 + L * S * z_curve) ** (1 / L)
            ax.plot(age_grid, y_curve, label=f"P{percentile}", linewidth=1.5)
        ax.scatter([age], [x], color="red", zorder=3, label="Medida atual")
        ax.set_title(f"Curvas de percentil — {bone}")
        ax.set_xlabel("Idade")
        ax.set_ylabel("Medida do osso")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, ncol=2)
        st.pyplot(fig)
    else:
        st.info("Calcule o z-score para ver a curva do percentil.")
