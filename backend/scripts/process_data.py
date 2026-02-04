import numpy as np
import pandas as pd


def process_dataset():
    input_path = "backend/data/source.csv"
    output_path = "backend/data/patients.csv"

    # Load the dataset
    try:
        df = pd.read_csv(input_path)
    except Exception:
        df = pd.read_csv(input_path, encoding="cp1251")

    # 1. Remove columns that are entirely empty (the trailing commas)
    df = df.dropna(axis=1, how="all")

    # 2. Rename Columns
    column_mapping = {
        "№": "patient_id",
        "пол": "gender",
        "Возраст": "age",
        "Стенокардия ФК": "angina_functional_class",
        "ПИКС": "post_infarction_cardiosclerosis",
        "МФА": "multifocal_atherosclerosis",
        "СД": "diabetes_mellitus",
        "ХОБЛ/БА": "copd_asthma",
        "Стенокардия через год": "angina_one_year_followup",
        "АГ": "hypertension",
        "Холестерин": "cholesterol_level",
        "ИМТ": "bmi",
        "ФВЛЖ, %": "lvef_percent",
        "Тип кровотока": "blood_flow_type",
        "ПНА прокс %": "lad_proximal_stenosis_percent",
        "ПНА сред %": "lad_mid_stenosis_percent",
        "ОА прокс %": "lcx_proximal_stenosis_percent",
        "ОА сред %": "lcx_mid_stenosis_percent",
        "ПКА прокс%": "rca_proximal_stenosis_percent",
        "ПКА сред %": "rca_mid_stenosis_percent",
        "Syntax Score": "syntax_score",
        "FFR": "ffr",
        r"DFR\iFR": "dfr_ifr",
        "Нестабильная бляшка": "unstable_plaque",
        "Объем бляшки, %": "plaque_volume_percent",
        "Просвет, мм2": "lumen_area_mm2",
        "Стент D": "stent_diameter",
        "Стент L": "stent_length",
        "Госпитальная смерть": "hospital_death",
        "Тромбоз стента": "stent_thrombosis",
        "Госпитальный ИМ": "hospital_mi",
        "ОНМК": "stroke_tia",
        "Инсульт": "stroke",
        "Повторная госпитализация": "repeated_hospitalization",
        "Повторная реваскуляризация": "repeated_revascularization",
        "ИМ": "myocardial_infarction_followup",
    }

    df = df.rename(columns=column_mapping)

    # 3. Clean Values

    # Gender
    df["gender"] = df["gender"].map({"м": "male", "ж": "female"})

    # Angina Follow-up (handling mixed numeric and string values)
    def clean_followup(val):
        if pd.isna(val):
            return val
        val = str(val).lower().strip()
        if val == "нет":
            return "none"
        if val == "смерть":
            return "death"
        if "чкв" in val:
            return val.replace("чкв", "PCI")
        return val

    if "angina_one_year_followup" in df.columns:
        df["angina_one_year_followup"] = df["angina_one_year_followup"].apply(
            clean_followup,
        )

    # Convert numeric columns that might have commas as decimal separators
    numeric_candidates = [
        "cholesterol_level",
        "bmi",
        "lvef_percent",
        "plaque_volume_percent",
        "lumen_area_mm2",
        "ffr",
        "dfr_ifr",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").replace("nan", np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill NaNs in binary outcome columns with 0 (assuming lack of entry means no event)
    outcome_cols = [
        "hospital_death",
        "stent_thrombosis",
        "hospital_mi",
        "stroke_tia",
        "stroke",
        "repeated_hospitalization",
        "repeated_revascularization",
        "myocardial_infarction_followup",
    ]
    for col in outcome_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 4. Save
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    process_dataset()
