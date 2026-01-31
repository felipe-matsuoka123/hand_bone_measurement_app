import streamlit as st
import pandas as pd
from scipy.stats import norm
import numpy as np
from src.utils import z_and_percentile
import matplotlib.pyplot as plt


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
