import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from typing import Optional


WHO_DATA_PATHS = {
    ("BMI", "Feminino"): "./data/bmi_f_who.csv",
    ("BMI", "Masculino"): "./data/bmi_m_who.csv",
    ("Altura para idade", "Feminino"): "./data/hfa_f_who.csv",
    ("Altura para idade", "Masculino"): "./data/hfa_m_who.csv",
}

CDC_DATA_PATHS = {
    ("Estatura para idade", "Feminino"): "./data/stat_f_cdc.csv",
    ("Estatura para idade", "Masculino"): "./data/stat_m_cdc.csv",
    ("Comprimento para idade", "Feminino"): "./data/len_f_cdc.csv",
    ("Comprimento para idade", "Masculino"): "./data/len_m_cdc.csv",
}

BODY_PROPORTION_DATA_PATHS = {
    ("Altura sentada para idade", "Feminino"): "./data/sh_f.csv",
    ("Altura sentada para idade", "Masculino"): "./data/sh_m.csv",
    ("Comprimento da perna para idade", "Feminino"): "./data/ll_f.csv",
    ("Comprimento da perna para idade", "Masculino"): "./data/ll_m.csv",
    ("Razão altura sentada/altura para idade", "Feminino"): "./data/sh_h_f.csv",
    ("Razão altura sentada/altura para idade", "Masculino"): "./data/sh_h_m.csv",
}

GROWTH_CHART_GROUPS = {
    "Curvas de crescimento WHO": {
        "label": "WHO",
        "paths": WHO_DATA_PATHS,
        "chart_types": ["BMI", "Altura para idade"],
    },
    "Curvas de crescimento CDC": {
        "label": "CDC",
        "paths": CDC_DATA_PATHS,
        "chart_types": ["Estatura para idade", "Comprimento para idade"],
    },
    "Relação estatura sentada/estatura": {
        "label": "Relação estatura sentada/estatura",
        "paths": BODY_PROPORTION_DATA_PATHS,
        "chart_types": [
            "Altura sentada para idade",
            "Comprimento da perna para idade",
            "Razão altura sentada/altura para idade",
        ],
    },
}

GROWTH_CHART_FIELD_LABELS = {
    "BMI": "BMI",
    "Altura para idade": "Altura (cm)",
    "Estatura para idade": "Estatura (cm)",
    "Comprimento para idade": "Comprimento (cm)",
    "Altura sentada para idade": "Altura sentada (cm)",
    "Comprimento da perna para idade": "Comprimento da perna (cm)",
    "Razão altura sentada/altura para idade": "Razão altura sentada/altura (ex.: 0.52)",
}


def lms_z(x: float, L: float, M: float, S: float) -> float:
    if x <= 0 or M <= 0 or S <= 0:
        return np.nan
    if abs(L) < 1e-12:
        return np.log(x / M) / S
    return (((x / M) ** L) - 1.0) / (L * S)


def lms_for_age(
    df: pd.DataFrame,
    age: float,
    bone: Optional[str] = None,
    sex: Optional[str] = None,
):
    sub = df.copy()
    if bone is not None and "Bone" in sub.columns:
        sub = sub[sub["Bone"] == bone]
    if sex is not None and "Sex" in sub.columns:
        sub = sub[sub["Sex"] == sex]
    sub = sub.sort_values("Age")

    if sub.empty:
        return np.nan, np.nan, np.nan

    ages = sub["Age"].to_numpy()
    ls_values = sub["L"].to_numpy()
    ms_values = sub["M"].to_numpy()
    ss_values = sub["S"].to_numpy()

    if age <= ages[0]:
        return ls_values[0], ms_values[0], ss_values[0]
    if age >= ages[-1]:
        return ls_values[-1], ms_values[-1], ss_values[-1]

    return (
        float(np.interp(age, ages, ls_values)),
        float(np.interp(age, ages, ms_values)),
        float(np.interp(age, ages, ss_values)),
    )


def z_and_percentile(
    df: pd.DataFrame,
    age: float,
    x: float,
    bone: Optional[str] = None,
    sex: Optional[str] = None,
):
    L, M, S = lms_for_age(df, age, bone=bone, sex=sex)
    z = lms_z(x, L, M, S)
    p = float(100 * norm.cdf(z)) if np.isfinite(z) else np.nan
    return float(z), p


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


def load_optional_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return load_csv(path)
    except FileNotFoundError:
        return None


def normalize_age_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Age" in df.columns:
        return df
    if "Month" in df.columns:
        renamed = df.rename(columns={"Month": "Age"}).copy()
        return renamed
    raise ValueError("O arquivo precisa ter uma coluna 'Age' ou 'Month'.")


def convert_age_to_data_scale(age_value: float, input_unit: str, data_unit: str) -> float:
    if input_unit == data_unit:
        return age_value
    if input_unit == "Anos" and data_unit == "Meses":
        return age_value * 12.0
    if input_unit == "Meses" and data_unit == "Anos":
        return age_value / 12.0
    return age_value


def percentile_curves(
    ax,
    df: pd.DataFrame,
    age: float,
    x: float,
    title: str,
    y_label: str,
    x_label: str,
):
    age_grid = np.linspace(df["Age"].min(), df["Age"].max(), 200)
    ordered = df.sort_values("Age")
    l_values = np.interp(age_grid, ordered["Age"], ordered["L"])
    m_values = np.interp(age_grid, ordered["Age"], ordered["M"])
    s_values = np.interp(age_grid, ordered["Age"], ordered["S"])

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for percentile in percentiles:
        z_curve = norm.ppf(percentile / 100.0)
        if abs(z_curve) < 1e-12:
            y_curve = m_values
        elif np.any(np.abs(l_values) < 1e-12):
            y_curve = m_values * np.exp(s_values * z_curve)
        else:
            y_curve = m_values * (1 + l_values * s_values * z_curve) ** (1 / l_values)
        ax.plot(age_grid, y_curve, label=f"P{percentile}", linewidth=1.5)

    ax.scatter([age], [x], color="red", zorder=3, label="Medida atual")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, ncol=2)


def adjust_x_axis_proportion(
    ax,
    df: pd.DataFrame,
    reference_span: float,
    current_age: Optional[float] = None,
):
    age_min = float(df["Age"].min())
    age_max = float(df["Age"].max())

    if current_age is not None and np.isfinite(current_age):
        age_min = min(age_min, float(current_age))
        age_max = max(age_max, float(current_age))

    age_span = max(age_max - age_min, 1e-6)

    ax.set_xlim(age_min, age_max)

    if reference_span <= 0:
        return

    width_fraction = max(0.45, min(age_span / reference_span, 1.0))
    left = 0.1
    bottom = 0.12
    max_width = 0.85
    height = 0.8
    ax.set_position([left, bottom, max_width * width_fraction, height])


def ensure_point_visible(
    ax,
    current_age: Optional[float],
    current_value: Optional[float],
):
    if current_age is not None and np.isfinite(current_age):
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(min(x_min, float(current_age)), max(x_max, float(current_age)))

    if current_value is not None and np.isfinite(current_value):
        y_min, y_max = ax.get_ylim()
        y_min = min(y_min, float(current_value))
        y_max = max(y_max, float(current_value))
        y_span = max(y_max - y_min, 1e-6)
        y_padding = y_span * 0.05
        ax.set_ylim(y_min - y_padding, y_max + y_padding)


st.title("Perfil metacarpofalangeano e curvas de crescimento")
st.caption("Cálculo do z-score e percentis com base em curvas LMS.")

bone_df = load_csv("./data/BoneMeasurements.csv")

chart_group = st.selectbox(
    "Tipo de avaliação",
    ["Medidas ósseas da mão", *GROWTH_CHART_GROUPS.keys()],
)

left_col, right_col = st.columns([1, 1.3])

with left_col:
    st.subheader("Adicione as medidas")
    submitted = False
    z = np.nan
    plot_df = None
    current_age = None
    current_value = None
    chart_title = ""
    y_label = ""
    x_label = "Idade"
    x_axis_reference_span = 1.0
    can_render_result = False

    if chart_group == "Medidas ósseas da mão":
        st.caption("Escolha o osso e informe a medida observada.")
        bone_age_unit = st.selectbox("Unidade da idade", ["Anos", "Meses"], key="bone_age_unit")
        with st.form("bone_params_form", clear_on_submit=False):
            sex = st.selectbox("Sexo", sorted(bone_df["Sex"].unique()))
            bone_age_label = "Idade (anos)" if bone_age_unit == "Anos" else "Idade (meses)"
            bone_age_max = 30.0 if bone_age_unit == "Anos" else 360.0
            bone_age_value = 10.0 if bone_age_unit == "Anos" else 120.0
            bone_age_step = 0.1 if bone_age_unit == "Anos" else 1.0
            age = st.number_input(
                bone_age_label,
                min_value=0.0,
                max_value=bone_age_max,
                value=bone_age_value,
                step=bone_age_step,
                help="Pode usar decimais com ponto ou vírgula, por exemplo: 14.5 ou 14,5.",
            )
            bone = st.selectbox("Osso", sorted(bone_df["Bone"].unique()))
            x = st.number_input(
                "Medida do osso",
                min_value=0.0,
                value=0.0,
                step=0.01,
                help="Pode usar decimais com ponto ou vírgula. Use a mesma unidade usada no estudo.",
            )
            submitted = st.form_submit_button("Calcular")

        if submitted:
            age_for_calc = convert_age_to_data_scale(age, bone_age_unit, "Anos")
            z, p = z_and_percentile(bone_df, age_for_calc, x, bone=bone, sex=sex)
            plot_df = (
                bone_df[(bone_df["Bone"] == bone) & (bone_df["Sex"] == sex)]
                .sort_values("Age")
                .copy()
            )
            current_age = age_for_calc
            current_value = x
            chart_title = f"Curvas de percentil - {bone}"
            y_label = "Medida do osso"
            x_label = "Idade"
            x_axis_reference_span = float(plot_df["Age"].max() - plot_df["Age"].min()) if not plot_df.empty else 1.0
            can_render_result = True

    else:
        growth_chart_config = GROWTH_CHART_GROUPS[chart_group]
        dataset_label = growth_chart_config["label"]
        st.caption(f"Escolha a curva {dataset_label} e informe a medida observada.")
        chart_type = st.selectbox("Curva", growth_chart_config["chart_types"])
        growth_age_unit = st.selectbox("Unidade da idade", ["Meses", "Anos"], key="growth_age_unit")
        with st.form("growth_params_form", clear_on_submit=False):
            sex = st.selectbox("Sexo", ["Feminino", "Masculino"])
            growth_age_label = "Idade (meses)" if growth_age_unit == "Meses" else "Idade (anos)"
            growth_age_max = 240.0 if growth_age_unit == "Meses" else 20.0
            growth_age_value = 60.0 if growth_age_unit == "Meses" else 5.0
            growth_age_step = 1.0 if growth_age_unit == "Meses" else 0.1
            age = st.number_input(
                growth_age_label,
                min_value=0.0,
                max_value=growth_age_max,
                value=growth_age_value,
                step=growth_age_step,
                help="Pode usar decimais com ponto ou vírgula, por exemplo: 14.5 ou 14,5.",
            )
            field_label = GROWTH_CHART_FIELD_LABELS[chart_type]
            x = st.number_input(
                field_label,
                min_value=0.0,
                value=0.0,
                step=0.01,
                help=(
                    f"Pode usar decimais com ponto ou vírgula. Use a mesma unidade adotada no CSV {dataset_label}."
                    " Para a razão altura sentada/altura, informe em formato decimal, por exemplo 0.52."
                ),
            )
            submitted = st.form_submit_button("Calcular")

        growth_path = growth_chart_config["paths"][(chart_type, sex)]
        growth_df = load_optional_csv(growth_path)

        if growth_df is None:
            st.warning(f"Arquivo ainda não encontrado: {growth_path}")
        else:
            try:
                growth_df = normalize_age_column(growth_df)
            except ValueError as exc:
                st.error(str(exc))
                growth_df = None

        if submitted and growth_df is not None:
            age_for_calc = convert_age_to_data_scale(age, growth_age_unit, "Meses")
            z, p = z_and_percentile(growth_df, age_for_calc, x)
            plot_df = growth_df.sort_values("Age").copy()
            current_age = age_for_calc
            current_value = x
            chart_title = f"Curvas {dataset_label} - {chart_type}"
            y_label = field_label
            x_label = "Idade (meses)"
            x_axis_reference_span = 240.0
            can_render_result = True

    if submitted and can_render_result:
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
    if submitted and can_render_result and plot_df is not None and np.isfinite(z):
        fig, ax = plt.subplots(figsize=(10, 7))
        percentile_curves(
            ax,
            plot_df,
            current_age,
            current_value,
            chart_title,
            y_label,
            x_label,
        )
        adjust_x_axis_proportion(ax, plot_df, x_axis_reference_span, current_age=current_age)
        ensure_point_visible(ax, current_age, current_value)
        st.pyplot(fig)
    else:
        st.info("Calcule o z-score para ver a curva do percentil.")
