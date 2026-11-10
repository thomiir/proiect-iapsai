import pandas as pd
import numpy as np

def assign_recommendations(row):
    """
    Generează teste recomandate și raționament clinic extins,
    incluzând tratarea cazurilor cu sex necunoscut în zona de graniță (12-13 g/dL).
    """
    pattern = row["pattern"]
    severity = row["severity"]
    mcv = row.get("mcv_fl", np.nan)
    rbc = row.get("rbc_m_ul", np.nan)
    hgb = row.get("hgb_g_dl", np.nan)
    sex = row.get("sex", np.nan)

    # Caz special: Sex necunoscut în zona de graniță 12.0 - 13.0 g/dL
    if pd.isna(sex) and pd.notna(hgb) and 12.0 <= hgb < 13.0:
        return (
            "repeat CBC; serum ferritin",
            f"Hb {hgb:.1f} g/dL is borderline: normal for adult female (>=12.0 g/dL), but indicates mild anemia if male (threshold 13.0 g/dL). Clinical correlation needed."
        )

    if pattern == "normal":
        return ("", "All core CBC parameters are within normal reference ranges.")

    # 1. Microcytic Hypochromic: discriminare carență de fier vs. trăsătură talasemică (Mentzer)
    if pattern == "microcytic hypochromic":
        if pd.notna(mcv) and pd.notna(rbc) and rbc > 0:
            mentzer = mcv / rbc
            if mentzer < 13.0:
                return (
                    "hemoglobin electrophoresis; ferritin; complete iron panel",
                    f"Marked microcytosis with relatively high RBC (Mentzer index: {mentzer:.1f} < 13); strongly suggestive of thalassemia trait."
                )
        return (
            "ferritin; serum iron; TIBC; transferrin saturation",
            "Microcytic hypochromic pattern; classic indication for iron deficiency anemia evaluation."
        )

    # 2. Macrocytic: discriminare severă vs. moderată
    if pattern == "macrocytic":
        if pd.notna(mcv) and mcv > 110.0:
            return (
                "vitamin B12; serum folate; reticulocyte count; peripheral blood smear; LDH",
                "Pronounced macrocytosis (MCV > 110 fL); high suspicion of megaloblastic anemia or myelodysplastic changes."
            )
        return (
            "vitamin B12; serum folate; reticulocyte count; TSH",
            "Macrocytic anemia; evaluate for B12/folate deficiency or secondary endocrine causes."
        )

    # 3. Normocytic: diferențiere severitate
    if pattern == "normocytic":
        if severity == "severe":
            return (
                "reticulocyte count; LDH; haptoglobin; direct Coombs test; urgent hematology consult",
                "Severe normocytic anemia; urgent diagnostic evaluation required for acute hemolysis or occult hemorrhage."
            )
        return (
            "reticulocyte count; serum creatinine; CRP; ferritin",
            "Normocytic anemia; evaluate renal function, systemic inflammation, or early nutritional deficiency."
        )

    return (
        "peripheral blood smear; hematology consult",
        "Atypical hematologic pattern; specialist consultation recommended."
    )


def enrich_cbc(df):
    numeric_cols = ["hgb_g_dl", "mcv_fl", "mch_pg", "mchc_g_dl", "rbc_m_ul"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flag HGB cu gestionare duală când sexul lipsește
    def flag_hgb(row):
        h = row.get("hgb_g_dl", np.nan)
        if pd.isna(h):
            return "unknown"
        sex = str(row.get("sex")).strip().upper()
        if sex == "M":
            return "low" if h < 13.0 else "normal"
        elif sex == "F":
            return "low" if h < 12.0 else "normal"
        else:
            # Sex necunoscut: <12.0 este low cert; 12.0-13.0 este borderline; >=13.0 este normal
            if h < 12.0:
                return "low"
            elif h < 13.0:
                return "borderline"
            return "normal"

    df["flags_hgb"] = df.apply(flag_hgb, axis=1)
    df["flags_mcv"] = np.where(df["mcv_fl"] < 80.0, "low", np.where(df["mcv_fl"] > 100.0, "high", "normal"))
    df["flags_mch"] = np.where(df["mch_pg"] < 27.0, "low", np.where(df["mch_pg"] > 33.0, "high", "normal"))
    df["flags_mchc"] = np.where(df["mchc_g_dl"] < 32.0, "low", np.where(df["mchc_g_dl"] > 36.0, "high", "normal"))

    # Morphological Pattern
    def pattern(row):
        if row["flags_hgb"] == "normal":
            return "normal"
        if row["flags_mcv"] == "low":
            return "microcytic hypochromic"
        elif row["flags_mcv"] == "high":
            return "macrocytic"
        return "normocytic"

    df["pattern"] = df.apply(pattern, axis=1)

    # Severity & Risk
    def severity(row):
        if row["flags_hgb"] == "normal":
            return "none"
        h = row["hgb_g_dl"]
        if pd.isna(h):
            return "none"
        if h >= 12.0:
            return "borderline"
        elif h >= 11.0:
            return "mild"
        elif h >= 8.0:
            return "moderate"
        return "severe"

    df["severity"] = df.apply(severity, axis=1)
    df["risk"] = df["severity"].map({
        "none": "low",
        "borderline": "low",
        "mild": "low",
        "moderate": "medium",
        "severe": "high"
    })

    # Asignare teste și rationale extins
    recs_rationales = df.apply(assign_recommendations, axis=1)
    df["recommended_tests"] = [r[0] for r in recs_rationales]
    df["rationale"] = [r[1] for r in recs_rationales]

    return df


def process_all_datasets():
    # ==========================================
    # 1. SURSA C: Mendeley Raw
    # ==========================================
    df_c = pd.read_csv("raw_data/mendeley_raw.csv")
    df_c.columns = df_c.columns.str.strip()

    hgb_col_c = next((c for c in df_c.columns if "HGB" in c.upper() or "HEM" in c.upper()), "HGB")
    mcv_col_c = next((c for c in df_c.columns if "MCV" in c.upper()), "MCV")
    mch_col_c = next((c for c in df_c.columns if "MCH" in c.upper() and "MCHC" not in c.upper()), "MCH")
    mchc_col_c = next((c for c in df_c.columns if "MCHC" in c.upper()), "MCHC")
    rbc_col_c = next((c for c in df_c.columns if "RBC" in c.upper()), "RBC")
    label_col_c = next((c for c in df_c.columns if "DECISION" in c.upper()), "Decision_Class")

    df_c = df_c.rename(columns={
        "Gender": "sex",
        hgb_col_c: "hgb_g_dl",
        mcv_col_c: "mcv_fl",
        mch_col_c: "mch_pg",
        mchc_col_c: "mchc_g_dl",
        rbc_col_c: "rbc_m_ul",
        label_col_c: "source_label"
    })

    if "sex" in df_c.columns:
        df_c["sex"] = df_c["sex"].astype(str).str.strip().str.upper()
        df_c["sex"] = df_c["sex"].replace({"0": "F", "1": "M", "FEMALE": "F", "MALE": "M"})
        df_c["sex"] = df_c["sex"].apply(lambda x: x if x in ["F", "M"] else np.nan)
    
    # Asigurare etichetă binară (0/1)
    df_c["source_label"] = pd.to_numeric(df_c["source_label"], errors="coerce").fillna(0).astype(int)
    df_c["original_diagnosis"] = df_c["source_label"].map({0: "Non-anemic", 1: "Anemic"})
    df_c["source"] = "mendeley_raw"

    # ==========================================
    # 2. SURSA B: Kaggle Types (Detectare robustă coloană text + mapare binară)
    # ==========================================
    df_b = pd.read_csv("raw_data/kaggle_types.csv")
    df_b.columns = df_b.columns.str.strip()

    # Detectează automat coloana cu textul diagnosticului
    type_col_b = next((c for c in df_b.columns if any(k in c.upper() for k in ["DIAGNOSIS"])), "Type")

    df_b = df_b.rename(columns={
        "HGB": "hgb_g_dl",
        "MCV": "mcv_fl",
        "MCH": "mch_pg",
        "MCHC": "mchc_g_dl",
        "RBC": "rbc_m_ul",
        type_col_b: "original_diagnosis"
    })
    
    # Mapare automată la 0 și 1: Healthy/Normal devine 0, restul bolilor devin 1
    def map_diagnosis_to_binary(val):
        if pd.isna(val):
            return 1
        s = str(val).strip().lower()
        if "healthy" in s or "normal" in s or s == "0":
            return 0
        return 1

    df_b["source_label"] = df_b["original_diagnosis"].apply(map_diagnosis_to_binary)
    df_b["source"] = "kaggle_types"
    df_b["sex"] = np.nan

    # ==========================================
    # 3. SURSA A: Kaggle Anemia (Rao)
    # ==========================================
    df_a = pd.read_csv("raw_data/kaggle_anemia.csv")
    df_a.columns = df_a.columns.str.strip()
    hgb_col_a = next((c for c in df_a.columns if "HEMO" in c.upper() or "HGB" in c.upper()), "Hemoglobin")
    label_col_a = next((c for c in df_a.columns if "RESULT" in c.upper() or "CLASS" in c.upper()), "Result")

    df_a = df_a.rename(columns={
        "Gender": "sex",
        hgb_col_a: "hgb_g_dl",
        "MCV": "mcv_fl",
        "MCH": "mch_pg",
        "MCHC": "mchc_g_dl",
        label_col_a: "source_label"
    })
    if "sex" in df_a.columns:
        df_a["sex"] = df_a["sex"].astype(str).str.strip().str.upper()
        df_a["sex"] = df_a["sex"].replace({"0": "F", "1": "M", "0.0": "F", "1.0": "M"})
        df_a["sex"] = df_a["sex"].apply(lambda x: x if x in ["F", "M"] else np.nan)

    df_a["source_label"] = pd.to_numeric(df_a["source_label"], errors="coerce").fillna(0).astype(int)
    df_a["original_diagnosis"] = df_a["source_label"].map({0: "Non-anemic", 1: "Anemic"})
    df_a["source"] = "kaggle_anemia"
    df_a["rbc_m_ul"] = np.nan

    # ==========================================
    # 4. CONCATENARE DOAR COLOANE STRICT COMUNE
    # ==========================================
    common_cols = [
        "source", "source_label", "original_diagnosis", "sex",
        "hgb_g_dl", "mcv_fl", "mch_pg", "mchc_g_dl", "rbc_m_ul"
    ]

    df_a = df_a[[c for c in common_cols if c in df_a.columns]]
    df_b = df_b[[c for c in common_cols if c in df_b.columns]]
    df_c = df_c[[c for c in common_cols if c in df_c.columns]]

    df_train = pd.concat([df_a, df_b, df_c], ignore_index=True)
    df_train = df_train.dropna(subset=["hgb_g_dl", "mcv_fl", "mch_pg", "mchc_g_dl"])
    df_train = df_train.drop_duplicates(subset=["hgb_g_dl", "mcv_fl", "mch_pg", "mchc_g_dl"])
    df_train["id"] = [f"train-{i+1:04d}" for i in range(len(df_train))]

    df_train = enrich_cbc(df_train)

    # ==========================================
    # 5. SET DE TEST (Shahane)
    # ==========================================
    df_test = pd.read_csv("raw_data/anemia_diagnosis_dataset.csv")
    df_test.columns = df_test.columns.str.strip()
    hgb_col_test = next((c for c in df_test.columns if any(k in c.upper() for k in ["HGB", "HB", "HEMO"])), "HGB")
    df_test = df_test.rename(columns={
        "Gender": "sex",
        "Sex": "sex",
        hgb_col_test: "hgb_g_dl",
        "MCV": "mcv_fl",
        "MCH": "mch_pg",
        "MCHC": "mchc_g_dl",
        "RBC": "rbc_m_ul"
    })
    if "sex" in df_test.columns:
        df_test["sex"] = df_test["sex"].astype(str).str.strip().str.upper()
        df_test["sex"] = df_test["sex"].replace({"0": "F", "1": "M", "0.0": "F", "1.0": "M"})
        df_test["sex"] = df_test["sex"].apply(lambda x: x if x in ["F", "M"] else np.nan)

    df_test["source"] = "shahane_test"
    df_test["source_label"] = np.nan
    df_test["original_diagnosis"] = "Unlabeled test data"
    df_test = df_test.dropna(subset=["hgb_g_dl", "mcv_fl", "mch_pg", "mchc_g_dl"])
    df_test["id"] = [f"test-{i+1:04d}" for i in range(len(df_test))]
    df_test = enrich_cbc(df_test)

    # ==========================================
    # 6. EXPORT FINAL
    # ==========================================
    final_cols = [
        "id", "source", "source_label", "original_diagnosis", "sex",
        "hgb_g_dl", "mcv_fl", "mch_pg", "mchc_g_dl", "rbc_m_ul",
        "flags_hgb", "flags_mcv", "flags_mch", "flags_mchc",
        "pattern", "severity", "risk",
        "recommended_tests", "rationale"
    ]

    df_train = df_train[final_cols]
    df_test = df_test[[c for c in final_cols if c in df_test.columns]]

    df_train.to_csv("data/train_unified.csv", index=False, encoding="utf-8")
    df_train.to_json("data/train_unified.jsonl", orient="records", lines=True, force_ascii=False)

    df_test.to_csv("data/test_unified.csv", index=False, encoding="utf-8")
    df_test.to_json("data/test_unified.jsonl", orient="records", lines=True, force_ascii=False)

    print("\n--- SUMAR TRAIN DATASET ---")
    print(df_train["source"].value_counts())
    print("\nDistribuție source_label (0/1):")
    print(df_train["source_label"].value_counts())
    print(f"\nTotal Train: {len(df_train)} rânduri | Total Test: {len(df_test)} rânduri")

if __name__ == "__main__":
    process_all_datasets()