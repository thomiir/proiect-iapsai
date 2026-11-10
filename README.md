# Data Proposal: Hemograma (CBC) -> tipar de anemie, risc și teste următoare

**Echipa:**
Baroană Alexandru-Cătălin, Crăciun Denis-Cristian, Jitescu Silviu-Cristian, Pricop Alexandru

## 1. De ce acest dataset?

Construim un set unificat și curățat de hemograme (CBC) din trei seturi publice de anemie (**$2.381$ înregistrări unice de antrenare**) și un set extern de evaluare (**$364$ înregistrări de test**), îmbogățit cu: flag-uri pe parametri (`low`/`borderline`/`normal`/`high`), severitate, tipar morfologic al anemiei, nivel de risc, recomandare de teste următoare și justificare clinică (`rationale`), în format CSV și JSONL.

**Cine l-ar folosi:** studenți și cercetători în ML (clasificare, fine-tuning de LLM pentru interpretarea analizelor), dezvoltatori de unelte educaționale de tip „asistent de laborator", studenți la medicină.

**Ce aduce în plus:** seturile existente conțin doar valori numerice și o etichetă simplă (`0`/`1`, `Decision_Class` sau tipul de anemie), cu denumiri de coloane și clase diferite, deci nu pot fi folosite împreună direct. Nu au flag-uri pe parametri, severitate uniformă, nivel de risc sau recomandări de investigații. Nu colectăm date noi din spitale: **unificăm, deduplicăm și îmbogățim** datele existente cu proveniență clară.

## 2. Surse

| Set | Link | Licență | Dimensiune brută $\rightarrow$ curățată | Câmpuri originale |
| --- | --- | --- | --- | --- |
| A. Anemia Dataset (Kaggle, B. R. Rao) | [https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset) | **Necunoscută** | $1.421 \rightarrow$ **$534$ rânduri unice** | Gender, Hemoglobin, MCH, MCHC, MCV, Result (0/1) |
| B. Anemia Types Classification (Kaggle, E. Aboelnaga) | [https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification](https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification) | **Apache 2.0** | $1.281 \rightarrow$ **$875$ rânduri unice** | HGB, RBC, WBC, PLT, MCV, MCH, MCHC, PDW, PCT, Diagnosis (9 clase) |
| C. Raw Hematological Dataset for Anemia Analysis and Classification (Mendeley, 2026) | [https://data.mendeley.com/datasets/tnswkb4bt8/1](https://data.mendeley.com/datasets/tnswkb4bt8/1) | **CC BY 4.0** | $1.005 \rightarrow$ **$972$ rânduri unice** | Gender, Age, HGB, RBC, PCV/HCT, MCV, MCH, MCHC, Decision_Class (0/1) |
| D. Anemia Diagnosis Dataset (Kaggle, Vohra / Shahane) — **Doar pentru Test** | [https://www.kaggle.com/datasets/saurabhshahane/anemia-diagnosis-dataset](https://www.kaggle.com/datasets/saurabhshahane/anemia-diagnosis-dataset) | **CC0: Public Domain** | **$364$ rânduri test** | Gender, Age, Hb, RBC, PCV, MCV, MCH, MCHC, RDW, TLC, PLT |

**Praguri de referință și reguli clinice:** Organizația Mondială a Sănătății (OMS) pentru hemoglobină ($13.0\text{ g/dL}$ bărbați, $12.0\text{ g/dL}$ femei) și Indicele Mentzer ($\text{MCV} / \text{RBC} < 13$) pentru discriminarea trăsăturii talasemice față de carența de fier.

**Licențe.** B, C și D permit redistribuirea cu atribuire. A nu are licență declarată, deci nu o redistribuim. **Nu urcăm date brute pe GitHub**: repository-ul conține scriptul reproductibil de procesare (`build_dataset.py`) și eșantionul extras din B și C.

## 3. Date sensibile

Toate sursele sunt complet anonimizate: fără nume, CNP, adrese sau alți identificatori personali. Pentru a evita matricele rare (sparse) și pentru a reduce cvasi-identificatorii, am eliminat vârsta și parametrii prezenți într-o singură sursă (`WBC`, `PLT`, `PCV`, `PDW`, `PCT`), păstrând strict sexul (`F`/`M`/`null`) și nucleul comun de parametri eritrocitari.

## 4. Schema

Am optat pentru o schemă plată ($19$ câmpuri), identică între `.csv` și `.jsonl`, standardizată în limba engleză pentru compatibilitate cu seturile originale și modelele LLM. Câmpurile `hgb_g_dl`, `mcv_fl`, `mch_pg`, `mchc_g_dl` sunt **obligatorii**; `sex` și `rbc_m_ul` sunt **opționale** (`null` dacă sursa nu le oferă).

```json
{
  "id": "string (ex: train-0001)",
  "source": "kaggle_anemia | kaggle_types | mendeley_raw | shahane_test",
  "source_label": "0 | 1 | null",
  "original_diagnosis": "string",
  "sex": "F | M | null",
  "hgb_g_dl": "float",
  "mcv_fl": "float",
  "mch_pg": "float",
  "mchc_g_dl": "float",
  "rbc_m_ul": "float | null",
  "flags_hgb": "low | borderline | normal",
  "flags_mcv": "low | normal | high",
  "flags_mch": "low | normal | high",
  "flags_mchc": "low | normal | high",
  "pattern": "normal | microcytic hypochromic | normocytic | macrocytic",
  "severity": "none | borderline | mild | moderate | severe",
  "risk": "low | medium | high",
  "recommended_tests": "string | null",
  "rationale": "string"
}
```

* **Unificarea etichetei sursă:** Diagnosticele text din sursa B au fost păstrate în `original_diagnosis` și mapate binar în `source_label` (`0` = Healthy/Non-anemic, `1` = Anemic), aliniindu-se cu sursele A și C.
* **Gestionarea sexului lipsă (Zona de graniță $12.0\text{--}13.0\text{ g/dL}$):** Când `sex` este `null` (în sursa B) și hemoglobina este între $12.0$ și $13.0\text{ g/dL}$, înregistrarea primește flag-ul și severitatea `borderline`, iar `rationale` explică dublul prag OMS (normal pentru femei $\ge 12.0\text{ g/dL}$, dar anemie ușoară pentru bărbați $< 13.0\text{ g/dL}$).

**Exemple reale din setul generat (`train_unified.jsonl`):**

```json
{"id":"train-1416","source":"mendeley_raw","source_label":0,"original_diagnosis":"Non-anemic","sex":"F","hgb_g_dl":11.5,"mcv_fl":78.0,"mch_pg":25.1,"mchc_g_dl":32.1,"rbc_m_ul":4.59,"flags_hgb":"low","flags_mcv":"low","flags_mch":"low","flags_mchc":"normal","pattern":"microcytic hypochromic","severity":"mild","risk":"low","recommended_tests":"ferritin; serum iron; TIBC; transferrin saturation","rationale":"Microcytic hypochromic pattern; classic indication for iron deficiency anemia evaluation."}
```

```json
{"id":"train-0618","source":"kaggle_types","source_label":1,"original_diagnosis":"Macrocytic anemia","sex":null,"hgb_g_dl":10.5,"mcv_fl":105.8,"mch_pg":33.8,"mchc_g_dl":32.1,"rbc_m_ul":3.1,"flags_hgb":"low","flags_mcv":"high","flags_mch":"high","flags_mchc":"normal","pattern":"macrocytic","severity":"moderate","risk":"medium","recommended_tests":"vitamin B12; serum folate; reticulocyte count; TSH","rationale":"Macrocytic anemia; evaluate for B12/folate deficiency or secondary endocrine causes."}
```

```json
{"id":"train-0542","source":"kaggle_types","source_label":1,"original_diagnosis":"Normocytic hypochromic anemia","sex":null,"hgb_g_dl":12.0,"mcv_fl":86.4,"mch_pg":27.3,"mchc_g_dl":31.6,"rbc_m_ul":4.39,"flags_hgb":"borderline","flags_mcv":"normal","flags_mch":"normal","flags_mchc":"low","pattern":"normocytic","severity":"borderline","risk":"low","recommended_tests":"repeat CBC; serum ferritin","rationale":"Hb 12.0 g/dL is borderline: normal for adult female (>=12.0 g/dL), but indicates mild anemia if male (threshold 13.0 g/dL). Clinical correlation needed."}
```

**Proveniență (rezumat):** `sex`, `hgb_g_dl`, `mcv_fl`, `mch_pg`, `mchc_g_dl`, `rbc_m_ul` și `original_diagnosis` sunt originale; `source_label` este mapat binar; `flags_*`, `pattern`, `severity`, `risk`, `recommended_tests` și `rationale` sunt derivate determinist prin reguli medicale (OMS, Indice Mentzer, praguri MCV/Hb).

## 5. (Opțional) Utilitate

Evaluăm un model LLM local (Llama 3.1 prin Ollama) pe setul extern de test (`test_unified.jsonl`, $364$ cazuri din sursa D), comparând capacitatea modelului de a prezice corect `pattern`, `risk` și `recommended_tests` înainte și după fine-tuning / few-shot prompting pe setul nostru îmbogățit.

**Eșantion:** `data/sample.csv`, $25\text{--}50$ de rânduri reale procesate din B și C, fără identificatori.
