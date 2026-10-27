# Data Proposal: Hemograma (CBC) -> tipar de anemie, risc și teste următoare

**Echipa:**
Baroană Alexandru-Cătălin
Crăciun Denis-Cristian
Jitescu Silviu-Cristian
Pricop Alexandru

## 1. De ce acest dataset?

Construim un set unificat de hemograme (CBC) din trei seturi publice de anemie, îmbogățit cu: flag-uri pe parametri (low/normal/high), severitate, tipar al anemiei, nivel de risc și recomandare de teste următoare, în JSON.

**Cine l-ar folosi:** studenți și cercetători în ML (clasificare, fine-tuning de LLM pentru interpretarea analizelor), dezvoltatori de unelte educaționale „asistent de laborator", studenți la medicină.

**Ce aduce în plus:** seturile existente au doar valori numerice și o etichetă (0/1, `Decision_Class` sau tipul de anemie), cu coloane și etichete diferite, deci nu se pot folosi împreună direct. Nu au flag-uri, severitate uniformă, risc sau recomandări. Nu colectăm date noi: **unificăm și îmbogățim**, cu proveniență clară. Un singur set existent nu ajunge: unul are sexul dar puțini parametri, altul are mulți parametri dar nu are sexul.

## 2. Surse

| Set | Link | Licență | Dimensiune | Câmpuri |
| --- | --- | --- | --- | --- |
| A. Anemia Dataset (Kaggle, B. R. Rao) | [https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset) | **Necunoscută** | ~1.421 rânduri | sex, HGB, MCH, MCHC, MCV, rezultat 0/1 |
| B. Anemia Types Classification (Kaggle, E. Aboelnaga) | [https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification](https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification) | **Apache 2.0** | ~1.281 rânduri | HGB, RBC, WBC, PLT, MCV, MCH, MCHC, PDW, PCT, tip anemie (9 clase) |
| C. Raw Hematological Dataset for Anemia Analysis and Classification (Mendeley, 2026) | [https://data.mendeley.com/datasets/tnswkb4bt8/1](https://data.mendeley.com/datasets/tnswkb4bt8/1) | **CC BY 4.0** | 1005 rânduri | Gender, Age, HGB, RBC, PCV, MCV, MCH, MCHC, Decision_Class |

*(Notă: Setul „Anemia Diagnosis Dataset” / Vohra et al. (https://www.kaggle.com/datasets/saurabhshahane/anemia-diagnosis-dataset) nu este inclus în setul de antrenament deoarece nu are coloană de rezultat; îl vom folosi exclusiv pentru teste/evaluare).*

Praguri de referință pentru Hb: OMS (reguli, nu date).

**Licențe.** B și C permit redistribuirea cu atribuire (păstrăm autori, link, licență). A nu are licență declarată, deci nu o redistribuim. **Nu urcăm date brute pe GitHub**: repo-ul conține scripturi de descărcare și transformare. Eșantionul vine din B și C. Dacă licența lui A nu se clarifică, versiunea publicată conține doar seturile cu licență clară (B, C).

## 3. Date sensibile

Toate sursele sunt anonimizate: fără nume, CNP, adrese sau alți identificatori; doar valori de laborator, sex și (în C) vârstă. Nu folosim date personale sau din spitale. Sexul și vârsta sunt cvasi-identificatori, deci le păstrăm doar ca valori simple și nu le combinăm cu surse externe.

## 4. Schema

Câmpurile comune tuturor seturilor (`hgb`, `mcv`, `mch`, `mchc`) sunt **obligatorii**; restul sunt **opționale** (`null` dacă sursa nu le are).

```json
{
  "id": "string",
  "source": "kaggle_anemia | kaggle_types | mendeley_raw",
  "patient": {"sex": "F|M|null", "age_years": "int|null"},
  "cbc": {
    "hgb_g_dl": "float", "mcv_fl": "float", "mch_pg": "float", "mchc_g_dl": "float",
    "rbc_m_ul": "float|null", "pcv_pct": "float|null", "wbc_k_ul": "float|null",
    "plt_k_ul": "float|null", "rdw_pct": "float|null", "pdw": "float|null", "pct": "float|null"
  },
  "flags": {"<parametru>": "low|normal|high"},
  "source_label": "string|null",
  "label": {
    "pattern": "normal|microcitară hipocromă|normocitară|macrocitară|alt tip",
    "severity": "none|mild|moderate|severe",
    "risk": "low|medium|high",
    "recommended_tests": ["string"],
    "rationale": "string"
  }
}

```

Etichetele originale diferă între surse, deci `pattern` și `severity` se **recalculează uniform** din valori; eticheta originală rămâne în `source_label`. Fără sex, flag-urile folosesc un interval conservator.

**Exemple** (ilustrative; eșantionul atașat conține rânduri reale):

```json
{"id":"ex-001","source":"mendeley_raw","patient":{"sex":"F","age_years":34},
 "cbc":{"hgb_g_dl":10.2,"mcv_fl":72,"mch_pg":23,"mchc_g_dl":31.5,"rbc_m_ul":4.4,"pcv_pct":32},
 "flags":{"hgb":"low","mcv":"low","mch":"low","mchc":"normal"},"source_label":"Anemia",
 "label":{"pattern":"microcitară hipocromă","severity":"moderate","risk":"medium",
  "recommended_tests":["feritină","fier seric","TIBC"],
  "rationale":"Hb, MCV și MCH scăzute: tipar compatibil cu carență de fier."}}

```

```json
{"id":"ex-002","source":"kaggle_anemia","patient":{"sex":"M","age_years":null},
 "cbc":{"hgb_g_dl":14.6,"mcv_fl":89,"mch_pg":30,"mchc_g_dl":34},
 "flags":{"hgb":"normal","mcv":"normal","mch":"normal","mchc":"normal"},"source_label":"0",
 "label":{"pattern":"normal","severity":"none","risk":"low","recommended_tests":[],
  "rationale":"Toți parametrii în intervalul de referință."}}

```

```json
{"id":"ex-003","source":"kaggle_types","patient":{"sex":null,"age_years":null},
 "cbc":{"hgb_g_dl":9.1,"mcv_fl":108,"mch_pg":34,"mchc_g_dl":33,"rbc_m_ul":2.9,"wbc_k_ul":5.8,"plt_k_ul":210},
 "flags":{"hgb":"low","mcv":"high","mch":"high","mchc":"normal"},"source_label":"Macrocytic anemia",
 "label":{"pattern":"macrocitară","severity":"moderate","risk":"medium",
  "recommended_tests":["vitamina B12","acid folic","reticulocite"],
  "rationale":"Hb scăzută cu MCV crescut: se investighează deficit de B12/folat."}}

```

**Proveniență (rezumat):** `patient`, `cbc`, `source_label` sunt originale; `flags` și `severity` sunt derivate din reguli (intervale de referință, praguri OMS); `pattern`, `risk`, `recommended_tests`, `rationale` sunt generate cu reguli și/sau LLM.

## 5. (Opțional) Utilitate

Comparăm un model mic (sau LLM) fine-tunat pe setul îmbogățit cu același model antrenat doar pe valori + etichetă brută, pe un set de test separat. Măsurăm acuratețea pe `pattern` și `risk`, plus acordul recomandărilor cu un eșantion de ~50 de înregistrări verificat manual față de ghiduri publice.

**Eșantion:** `data/sample.csv`, 20–50 de rânduri reale din B și C, fără identificatori.