import numpy as np
import pandas as pd
from scipy.stats import norm

def lms_z(x: float, L: float, M: float, S: float) -> float:
    if x <= 0 or M <= 0 or S <= 0:
        return np.nan
    if abs(L) < 1e-12:
        return np.log(x / M) / S
    return (((x / M) ** L) - 1.0) / (L * S)

def lms_for_age(df: pd.DataFrame, bone: str, sex: str, age: float):
    sub = (
        df[(df["Bone"] == bone) & (df["Sex"] == sex)]
        .sort_values("Age")
    )

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
