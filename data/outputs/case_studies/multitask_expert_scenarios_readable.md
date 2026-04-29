# Multitask expert scenarios (GPT-4o-mini vs Gemma4, prompt v3)

- **Count:** 24 scenarios
- **Primary:** GPT-4o-mini multitask zero-shot (`llm_zero_shot_results.csv`)
- **Comparison:** Gemma4 multitask zero-shot (`responses.jsonl`)
- **Machine-readable:** `multitask_expert_scenarios.jsonl`

## mt_expert_001 — pair_id 42

- **Tags:** both_models_same_wrong_tasks, gemma_parse_salvage, gemma_partial_output
- **Disagreeing tasks (0):** —
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 1, 'obesity': 0, 'cardio': 0, 'kidney': 1, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 1, 'prob': 1.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.75}, 'kidney': {'pred': 1, 'prob': 1.0}, 'stroke': {'pred': 0, 'prob': 0.0}}`

**Reasoning:**

```
The patient has a diagnosis of hypertensive chronic kidney disease and end stage renal disease, indicating current kidney issues. There is no evidence of lipid disorders, diabetes, obesity, or stroke. Hypertension is confirmed by the diagnosis. Cardiovascular issues are likely given the presence of cardiomyopathy and peripheral vascular diseases.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 1, 'prob': 0.8}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.6}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 40301;5856;4254;4239;7100;58381;2767;78701;78791;2753;28529;78904;33829;2875;44389;V0254;V5861;V4512;V4511;V1581. Current-visit narrative (truncated): - Diagnoses made: Hypertensive chronic kidney disease, malignant, with chronic kidney disease stage V or end stage renal disease; End stage renal disease; Other primary cardiomyopathies; Unspecified disease of pericardium; Systemic lupus erythematosus; Nephritis and nephropathy, not specified as acute or chronic, in diseases classified elsewhere; Hyperpotassemia; Nausea with vomiting; Diarrhea; Disorders of phosphorus metabolism; Anemia of other chronic disease; Abdominal pain, left lower quadrant; Other chronic pain; Thrombocytopenia, unspecified; Other specified peripheral vascular diseases; Carrier or suspected carrier of Methicillin resistant Staphylococcus aureus; Long-term (current) use of anticoagulants; Noncompliance with renal dialysis; Renal dialysis status; Personal history of noncompliance with medical treatment, presenting hazards to health - Medications prescribed: Cloni...

### Current-visit narrative

```
- Diagnoses made: Hypertensive chronic kidney disease, malignant, with chronic kidney disease stage V or end stage renal disease; End stage renal disease; Other primary cardiomyopathies; Unspecified disease of pericardium; Systemic lupus erythematosus; Nephritis and nephropathy, not specified as acute or chronic, in diseases classified elsewhere; Hyperpotassemia; Nausea with vomiting; Diarrhea; Disorders of phosphorus metabolism; Anemia of other chronic disease; Abdominal pain, left lower quadrant; Other chronic pain; Thrombocytopenia, unspecified; Other specified peripheral vascular diseases; Carrier or suspected carrier of Methicillin resistant Staphylococcus aureus; Long-term (current) use of anticoagulants; Noncompliance with renal dialysis; Renal dialysis status; Personal history of noncompliance with medical treatment, presenting hazards to health
- Medications prescribed: Clonidine Patch 0.1 mg/24 hr; Warfarin; Prochlorperazine; Labetalol; HydrALAzine; 5% Dextrose; PredniSONE; NIFEdipine CR; Aliskiren; Fentanyl Patch; Heparin; HYDROmorphone (Dilaudid); Ondansetron; Citalopram Hydrobromide; 0.9% Sodium Chloride; Heparin Flush (100 units/ml); Sodium Chloride 0.9%  Flush; Senna; Docusate Sodium; Clonidine Patch 0.3 mg/24 hr
- Procedures performed: Hemodialysis
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 42,
  "diagnoses_codes_current": "40301;5856;4254;4239;7100;58381;2767;78701;78791;2753;28529;78904;33829;2875;44389;V0254;V5861;V4512;V4511;V1581",
  "procedures_codes_current": "3995",
  "medications_current": "Clonidine Patch 0.1 mg/24 hr;Warfarin;Prochlorperazine;Labetalol;HydrALAzine;5% Dextrose;PredniSONE;NIFEdipine CR;Aliskiren;Fentanyl Patch;Heparin;HYDROmorphone (Dilaudid);Ondansetron;Citalopram Hydrobromide;0.9% Sodium Chloride;Heparin Flush (100 units/ml);Sodium Chloride 0.9%  Flush;Senna;Docusate Sodium;Clonidine Patch 0.3 mg/24 hr",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 1,
  "label_obesity_current": 0,
  "label_cardio_next": 0,
  "label_kidney_next": 1,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Hypertensive chronic kidney disease, malignant, with chronic kidney disease stage V or end stage renal disease; End stage renal disease; Other primary cardiomyopathies; Unspecified disease of pericardium; Systemic lupus erythematosus; Nephritis and nephropathy, not specified as acute or chronic, in diseases classified elsewhere; Hyperpotassemia; Nausea with vomiting; Diarrhea; Disorders of phosphorus metabolism; Anemia of other chronic disease; Abdominal pain, left lower quadrant; Other chronic pain; Thrombocytopenia, unspecified; Other specified peripheral vascular diseases; Carrier or suspected carrier of Methicillin resistant Staphylococcus aureus; Long-term (current) use of anticoagulants; Noncompliance with renal dialysis; Renal dialysis status; Personal history of noncompliance with medical treatment, presenting hazards to health\n- Medications prescribed: Clonidine Patch 0.1 mg/24 hr; Warfarin; Prochlorperazine; Labetalol; HydrALAzine; 5% Dextrose; PredniSONE; NIFEdipine CR; Aliskiren; Fentanyl Patch; Heparin; HYDROmorphone (Dilaudid); Ondansetron; Citalopram Hydrobromide; 0.9% Sodium Chloride; Heparin Flush (100 units/ml); Sodium Chloride 0.9%  Flush; Senna; Docusate Sodium; Clonidine Patch 0.3 mg/24 hr\n- Procedures performed: Hemodialysis"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_002 — pair_id 44

- **Tags:** both_models_same_wrong_tasks, gemma_parse_salvage, gemma_partial_output
- **Disagreeing tasks (0):** —
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 1, 'obesity': 0, 'cardio': 0, 'kidney': 1, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 1, 'prob': 1.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.75}, 'kidney': {'pred': 1, 'prob': 1.0}, 'stroke': {'pred': 0, 'prob': 0.0}}`

**Reasoning:**

```
The patient has a diagnosis of hypertensive chronic kidney disease and end stage renal disease, indicating current hypertension and kidney disease. There is no evidence of diabetes or obesity in the record. There are multiple cardiac-related diagnoses, suggesting a likelihood of cardiovascular disease, but no explicit evidence of lipid disorders or stroke.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 1, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 2766;5856;40301;4254;4239;28529;2767;7100;78909;33829;4589;53190;34590;311;V1251;V4512. Current-visit narrative (truncated): - Diagnoses made: [ICD9-DX:2766]; End stage renal disease; Hypertensive chronic kidney disease, malignant, with chronic kidney disease stage V or end stage renal disease; Other primary cardiomyopathies; Unspecified disease of pericardium; Anemia of other chronic disease; Hyperpotassemia; Systemic lupus erythematosus; Abdominal pain, other specified site; Other chronic pain; Hypotension, unspecified; Gastric ulcer, unspecified as acute or chronic, without mention of hemorrhage or perforation, without mention of obstruction; Epilepsy, unspecified, without mention of intractable epilepsy; Depressive disorder, not elsewhere classified; Personal history of venous thrombosis and embolism; Noncompliance with renal dialysis - Medications prescribed: Heparin; Magnesium Sulfate; SW; Gabapentin; Sodium Chloride 0.9% Flush; LeVETiracetam; Warfarin; Clonidine Patch 0.1 mg/24 hr; Clonidine Patch 0....

### Current-visit narrative

```
- Diagnoses made: [ICD9-DX:2766]; End stage renal disease; Hypertensive chronic kidney disease, malignant, with chronic kidney disease stage V or end stage renal disease; Other primary cardiomyopathies; Unspecified disease of pericardium; Anemia of other chronic disease; Hyperpotassemia; Systemic lupus erythematosus; Abdominal pain, other specified site; Other chronic pain; Hypotension, unspecified; Gastric ulcer, unspecified as acute or chronic, without mention of hemorrhage or perforation, without mention of obstruction; Epilepsy, unspecified, without mention of intractable epilepsy; Depressive disorder, not elsewhere classified; Personal history of venous thrombosis and embolism; Noncompliance with renal dialysis
- Medications prescribed: Heparin; Magnesium Sulfate; SW; Gabapentin; Sodium Chloride 0.9%  Flush; LeVETiracetam; Warfarin; Clonidine Patch 0.1 mg/24 hr; Clonidine Patch 0.3 mg/24 hr; Lidocaine 5% Patch; Calcium Gluconate; NIFEdipine CR; Labetalol; 0.9% Sodium Chloride; 5% Dextrose; HydrALAzine; Calcium Carbonate; HYDROmorphone (Dilaudid); sevelamer HYDROCHLORIDE; Aliskiren; Fentanyl Patch; PredniSONE; Citalopram Hydrobromide; Pantoprazole
- Procedures performed: Hemodialysis; Venous catheterization, not elsewhere classified; Transfusion of packed cells
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 44,
  "diagnoses_codes_current": "2766;5856;40301;4254;4239;28529;2767;7100;78909;33829;4589;53190;34590;311;V1251;V4512",
  "procedures_codes_current": "3995;3893;9904",
  "medications_current": "Heparin;Magnesium Sulfate;SW;Gabapentin;Sodium Chloride 0.9%  Flush;LeVETiracetam;Warfarin;Clonidine Patch 0.1 mg/24 hr;Clonidine Patch 0.3 mg/24 hr;Lidocaine 5% Patch;Calcium Gluconate;NIFEdipine CR;Labetalol;0.9% Sodium Chloride;5% Dextrose;HydrALAzine;Calcium Carbonate;HYDROmorphone (Dilaudid);sevelamer HYDROCHLORIDE;Aliskiren;Fentanyl Patch;PredniSONE;Citalopram Hydrobromide;Pantoprazole",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 1,
  "label_obesity_current": 0,
  "label_cardio_next": 0,
  "label_kidney_next": 1,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: [ICD9-DX:2766]; End stage renal disease; Hypertensive chronic kidney disease, malignant, with chronic kidney disease stage V or end stage renal disease; Other primary cardiomyopathies; Unspecified disease of pericardium; Anemia of other chronic disease; Hyperpotassemia; Systemic lupus erythematosus; Abdominal pain, other specified site; Other chronic pain; Hypotension, unspecified; Gastric ulcer, unspecified as acute or chronic, without mention of hemorrhage or perforation, without mention of obstruction; Epilepsy, unspecified, without mention of intractable epilepsy; Depressive disorder, not elsewhere classified; Personal history of venous thrombosis and embolism; Noncompliance with renal dialysis\n- Medications prescribed: Heparin; Magnesium Sulfate; SW; Gabapentin; Sodium Chloride 0.9%  Flush; LeVETiracetam; Warfarin; Clonidine Patch 0.1 mg/24 hr; Clonidine Patch 0.3 mg/24 hr; Lidocaine 5% Patch; Calcium Gluconate; NIFEdipine CR; Labetalol; 0.9% Sodium Chloride; 5% Dextrose; HydrALAzine; Calcium Carbonate; HYDROmorphone (Dilaudid); sevelamer HYDROCHLORIDE; Aliskiren; Fentanyl Patch; PredniSONE; Citalopram Hydrobromide; Pantoprazole\n- Procedures performed: Hemodialysis; Venous catheterization, not elsewhere classified; Transfusion of packed cells"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_003 — pair_id 58

- **Tags:** cross_model_disagreement, cross_model_disagreement_high, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (4):** lipid, diabetes, hypertension, cardio
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 0, 'obesity': 0, 'cardio': 1, 'kidney': 0, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['lipid', 'diabetes', 'hypertension', 'kidney']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.8}, 'diabetes': {'pred': 1, 'prob': 0.7}, 'hypertension': {'pred': 1, 'prob': 0.75}, 'obesity': {'pred': 0, 'prob': 0.1}, 'cardio': {'pred': 1, 'prob': 0.85}, 'kidney': {'pred': 1, 'prob': 0.9}, 'stroke': {'pred': 0, 'prob': 0.2}}`

**Reasoning:**

```
The patient has a history of coronary atherosclerosis and heart failure, which supports the prediction of cardiovascular disease. The presence of atorvastatin suggests a lipid disorder, and the use of insulin indicates diabetes. Hypertension is likely given the heart conditions. There is no evidence of obesity in the record. The patient has a malignant neoplasm of the kidney, which increases the likelihood of kidney disease. There is no indication of stroke risk in the current visit.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 1890;42820;4280;2874;9998;6823;5997;4148;V433;V4502;41401;78057;E9342;79092;E8798. Current-visit narrative (truncated): - Diagnoses made: Malignant neoplasm of kidney, except pelvis; Systolic heart failure, unspecified; Congestive heart failure, unspecified; [ICD9-DX:2874]; [ICD9-DX:9998]; Cellulitis and abscess of upper arm and forearm; [ICD9-DX:5997]; Other specified forms of chronic ischemic heart disease; Heart valve replaced by other means; Automatic implantable cardiac defibrillator in situ; Coronary atherosclerosis of native coronary artery; Unspecified sleep apnea; Anticoagulants causing adverse effects in therapeutic use; Abnormal coagulation profile; Other specified procedures as the cause of abnormal reaction of patient, or of later complication, without mention of misadventure at time of procedure - Medications prescribed: Acetaminophen; NS; Unasyn; Warfarin; Furosemide; Lorazepam; Pantoprazole Sodium; Docusate Sodium; Hydromorphone; Docusate Sodium (Liquid); Phenaseptic Throat Spray; Potas...

### Current-visit narrative

```
- Diagnoses made: Malignant neoplasm of kidney, except pelvis; Systolic heart failure, unspecified; Congestive heart failure, unspecified; [ICD9-DX:2874]; [ICD9-DX:9998]; Cellulitis and abscess of upper arm and forearm; [ICD9-DX:5997]; Other specified forms of chronic ischemic heart disease; Heart valve replaced by other means; Automatic implantable cardiac defibrillator in situ; Coronary atherosclerosis of native coronary artery; Unspecified sleep apnea; Anticoagulants causing adverse effects in therapeutic use; Abnormal coagulation profile; Other specified procedures as the cause of abnormal reaction of patient, or of later complication, without mention of misadventure at time of procedure
- Medications prescribed: Acetaminophen; NS; Unasyn; Warfarin; Furosemide; Lorazepam; Pantoprazole Sodium; Docusate Sodium; Hydromorphone; Docusate Sodium (Liquid); Phenaseptic Throat Spray; Potassium Chloride; SW; D5 1/2NS; Albuterol 0.083% Neb Soln; Ipratropium Bromide Neb; Sarna Lotion; Sodium Chloride 0.9%  Flush; Metoprolol; Oxycodone-Acetaminophen; Zolpidem Tartrate; Magnesium Sulfate; Metoprolol XL; Atorvastatin; Pantoprazole; Trandolapril; Oxycodone; Heparin Sodium; D5W; Argatroban; Acetylcysteine 20%; Sodium Bicarbonate; Cephalexin; Ampicillin Sodium; Magnesium Citrate; Fentanyl Citrate; Soln.; Propofol; Gentamicin; LR; Isotonic Sodium Chloride; Calcium Gluconate; Insulin
- Procedures performed: Partial nephrectomy; Pulmonary artery wedge monitoring; Diagnostic ultrasound of heart; Artificial pacemaker rate check; Transfusion of packed cells
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 58,
  "diagnoses_codes_current": "1890;42820;4280;2874;9998;6823;5997;4148;V433;V4502;41401;78057;E9342;79092;E8798",
  "procedures_codes_current": "554;8964;8872;8945;9904",
  "medications_current": "Acetaminophen;NS;Unasyn;Warfarin;Furosemide;Lorazepam;Pantoprazole Sodium;Docusate Sodium;Hydromorphone;Docusate Sodium (Liquid);Phenaseptic Throat Spray;Potassium Chloride;SW;D5 1/2NS;Albuterol 0.083% Neb Soln;Ipratropium Bromide Neb;Sarna Lotion;Sodium Chloride 0.9%  Flush;Metoprolol;Oxycodone-Acetaminophen;Zolpidem Tartrate;Magnesium Sulfate;Metoprolol XL;Atorvastatin;Pantoprazole;Trandolapril;Oxycodone;Heparin Sodium;D5W;Argatroban;Acetylcysteine 20%;Sodium Bicarbonate;Cephalexin;Ampicillin Sodium;Magnesium Citrate;Fentanyl Citrate;Soln.;Propofol;Gentamicin;LR;Isotonic Sodium Chloride;Calcium Gluconate;Insulin",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 1,
  "label_kidney_next": 0,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Malignant neoplasm of kidney, except pelvis; Systolic heart failure, unspecified; Congestive heart failure, unspecified; [ICD9-DX:2874]; [ICD9-DX:9998]; Cellulitis and abscess of upper arm and forearm; [ICD9-DX:5997]; Other specified forms of chronic ischemic heart disease; Heart valve replaced by other means; Automatic implantable cardiac defibrillator in situ; Coronary atherosclerosis of native coronary artery; Unspecified sleep apnea; Anticoagulants causing adverse effects in therapeutic use; Abnormal coagulation profile; Other specified procedures as the cause of abnormal reaction of patient, or of later complication, without mention of misadventure at time of procedure\n- Medications prescribed: Acetaminophen; NS; Unasyn; Warfarin; Furosemide; Lorazepam; Pantoprazole Sodium; Docusate Sodium; Hydromorphone; Docusate Sodium (Liquid); Phenaseptic Throat Spray; Potassium Chloride; SW; D5 1/2NS; Albuterol 0.083% Neb Soln; Ipratropium Bromide Neb; Sarna Lotion; Sodium Chloride 0.9%  Flush; Metoprolol; Oxycodone-Acetaminophen; Zolpidem Tartrate; Magnesium Sulfate; Metoprolol XL; Atorvastatin; Pantoprazole; Trandolapril; Oxycodone; Heparin Sodium; D5W; Argatroban; Acetylcysteine 20%; Sodium Bicarbonate; Cephalexin; Ampicillin Sodium; Magnesium Citrate; Fentanyl Citrate; Soln.; Propofol; Gentamicin; LR; Isotonic Sodium Chloride; Calcium Gluconate; Insulin\n- Procedures performed: Partial nephrectomy; Pulmonary artery wedge monitoring; Diagnostic ultrasound of heart; Artificial pacemaker rate check; Transfusion of packed cells"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_004 — pair_id 94

- **Tags:** both_models_same_wrong_tasks, gemma_parse_salvage, gemma_partial_output
- **Disagreeing tasks (0):** —
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 1, 'hypertension': 0, 'obesity': 0, 'cardio': 0, 'kidney': 0, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 1, 'prob': 1.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.75}, 'kidney': {'pred': 0, 'prob': 0.0}, 'stroke': {'pred': 0, 'prob': 0.0}}`

**Reasoning:**

```
The patient has a documented diagnosis of diabetes with neurological manifestations and is on long-term insulin. There is no evidence of hypertension or obesity in the current visit record. The presence of atherosclerosis and congestive heart failure suggests a higher risk for cardiovascular disease. There is no indication of kidney disease or stroke in the current visit.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 1, 'prob': 0.95}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.95}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 51884;5070;5100;44024;70703;5118;4280;5121;27652;42731;5990;24290;486;V4975;V5865;25060;3572;V5867;4920;99881;1121;0414. Current-visit narrative (truncated): - Diagnoses made: Acute and chronic respiratory failure; Pneumonitis due to inhalation of food or vomitus; Empyema with fistula; Atherosclerosis of native arteries of the extremities with gangrene; Pressure ulcer, lower back; [ICD9-DX:5118]; Congestive heart failure, unspecified; Iatrogenic pneumothorax; Hypovolemia; Atrial fibrillation; Urinary tract infection, site not specified; Thyrotoxicosis without mention of goiter or other cause, and without mention of thyrotoxic crisis or storm; Pneumonia, organism unspecified; Below knee amputation status; Long-term (current) use of steroids; Diabetes with neurological manifestations, type II or unspecified type, not stated as uncontrolled; Polyneuropathy in diabetes; Long-term (current) use of insulin; Emphysematous bleb; Emphysema (subcutaneous) (surgical) resulting from procedure; Candidiasis of vulva and vagina; [ICD9-DX:0414] - Medicati...

### Current-visit narrative

```
- Diagnoses made: Acute and chronic respiratory failure; Pneumonitis due to inhalation of food or vomitus; Empyema with fistula; Atherosclerosis of native arteries of the extremities with gangrene; Pressure ulcer, lower back; [ICD9-DX:5118]; Congestive heart failure, unspecified; Iatrogenic pneumothorax; Hypovolemia; Atrial fibrillation; Urinary tract infection, site not specified; Thyrotoxicosis without mention of goiter or other cause, and without mention of thyrotoxic crisis or storm; Pneumonia, organism unspecified; Below knee amputation status; Long-term (current) use of steroids; Diabetes with neurological manifestations, type II or unspecified type, not stated as uncontrolled; Polyneuropathy in diabetes; Long-term (current) use of insulin; Emphysematous bleb; Emphysema (subcutaneous) (surgical) resulting from procedure; Candidiasis of vulva and vagina; [ICD9-DX:0414]
- Medications prescribed: D5W; Fentanyl Citrate; Midazolam HCl; Piperacillin-Tazobactam Na; Iso-Osmotic Dextrose; Docusate Sodium (Liquid); Metoprolol; Diltiazem; Heparin; Aspirin; Sodium Chloride 0.9%  Flush; Lansoprazole Oral Suspension; Senna; Acetaminophen; Accolate; Clopidogrel Bisulfate; Methimazole; Insulin; Levothyroxine Sodium; Potassium Chloride; Furosemide; Vial; Gabapentin; Haloperidol; Nortriptyline; Dextrose 50%; Lorazepam; Ciprofloxacin HCl; Meropenem; NS (Mini Bag Plus); Ferrous Sulfate; Olanzapine; Sodium Polystyrene Sulfonate; Vancomycin HCl; Methylprednisolone Na Succ; Fluticasone Propionate 110mcg; Albuterol; Prednisone; Olanzapine (Disintegrating Tablet); Azithromycin; Insulin Human Regular; NS; Ipratropium Bromide MDI; Methylprednisolone Na Succ.; Syringe (IV Room); Metolazone; Alteplase (Catheter Clearance); Oxymetazoline HCl; traZODONE HCl; Zolpidem Tartrate
- Procedures performed: Temporary tracheostomy; Percutaneous [endoscopic] gastrostomy [PEG]; Other intubation of respiratory tract; Non-invasive mechanical ventilation; Insertion of endotracheal tube; Continuous invasive mechanical ventilation for 96 consecutive hours or more; Injection into thoracic cavity; Injection of antibiotic; Enteral infusion of concentrated nutritional substances; Venous catheterization, not elsewhere classified; Insertion of intercostal catheter for drainage; Transfusion of packed cells; Closed [endoscopic] biopsy of bronchus
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 94,
  "diagnoses_codes_current": "51884;5070;5100;44024;70703;5118;4280;5121;27652;42731;5990;24290;486;V4975;V5865;25060;3572;V5867;4920;99881;1121;0414",
  "procedures_codes_current": "311;4311;9605;9390;9604;9672;3492;9921;966;3893;3404;9904;3324",
  "medications_current": "D5W;Fentanyl Citrate;Midazolam HCl;Piperacillin-Tazobactam Na;Iso-Osmotic Dextrose;Docusate Sodium (Liquid);Metoprolol;Diltiazem;Heparin;Aspirin;Sodium Chloride 0.9%  Flush;Lansoprazole Oral Suspension;Senna;Acetaminophen;Accolate;Clopidogrel Bisulfate;Methimazole;Insulin;Levothyroxine Sodium;Potassium Chloride;Furosemide;Vial;Gabapentin;Haloperidol;Nortriptyline;Dextrose 50%;Lorazepam;Ciprofloxacin HCl;Meropenem;NS (Mini Bag Plus);Ferrous Sulfate;Olanzapine;Sodium Polystyrene Sulfonate;Vancomycin HCl;Methylprednisolone Na Succ;Fluticasone Propionate 110mcg;Albuterol;Prednisone;Olanzapine (Disintegrating Tablet);Azithromycin;Insulin Human Regular;NS;Ipratropium Bromide MDI;Methylprednisolone Na Succ.;Syringe (IV Room);Metolazone;Alteplase (Catheter Clearance);Oxymetazoline HCl;traZODONE HCl;Zolpidem Tartrate;Esmolol;Amiodarone HCl;Ascorbic Acid;Lactulose;AcetaZOLAMIDE Sodium;traMADOL;D5 1/2NS;Cosyntropin;Magnesium Sulfate;Heparin Sodium;Salmeterol Xinafoate Diskus (50 mcg);Metronidazole;CefTRIAXone;Propofol;Ipratropium Bromide Neb;Neutra-Phos;Zinc Sulfate;Miconazole Powder 2%;Fluconazole;Chlorhexidine Gluconate;Cefazolin;Heparin Flush PICC (100 units/ml);Doxycycline Hyclate;Syringe (Intrapleural);Morphine Sulfate;SW",
  "label_lipid_next": 0,
  "label_diabetes_current": 1,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 0,
  "label_kidney_next": 0,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Acute and chronic respiratory failure; Pneumonitis due to inhalation of food or vomitus; Empyema with fistula; Atherosclerosis of native arteries of the extremities with gangrene; Pressure ulcer, lower back; [ICD9-DX:5118]; Congestive heart failure, unspecified; Iatrogenic pneumothorax; Hypovolemia; Atrial fibrillation; Urinary tract infection, site not specified; Thyrotoxicosis without mention of goiter or other cause, and without mention of thyrotoxic crisis or storm; Pneumonia, organism unspecified; Below knee amputation status; Long-term (current) use of steroids; Diabetes with neurological manifestations, type II or unspecified type, not stated as uncontrolled; Polyneuropathy in diabetes; Long-term (current) use of insulin; Emphysematous bleb; Emphysema (subcutaneous) (surgical) resulting from procedure; Candidiasis of vulva and vagina; [ICD9-DX:0414]\n- Medications prescribed: D5W; Fentanyl Citrate; Midazolam HCl; Piperacillin-Tazobactam Na; Iso-Osmotic Dextrose; Docusate Sodium (Liquid); Metoprolol; Diltiazem; Heparin; Aspirin; Sodium Chloride 0.9%  Flush; Lansoprazole Oral Suspension; Senna; Acetaminophen; Accolate; Clopidogrel Bisulfate; Methimazole; Insulin; Levothyroxine Sodium; Potassium Chloride; Furosemide; Vial; Gabapentin; Haloperidol; Nortriptyline; Dextrose 50%; Lorazepam; Ciprofloxacin HCl; Meropenem; NS (Mini Bag Plus); Ferrous Sulfate; Olanzapine; Sodium Polystyrene Sulfonate; Vancomycin HCl; Methylprednisolone Na Succ; Fluticasone Propionate 110mcg; Albuterol; Prednisone; Olanzapine (Disintegrating Tablet); Azithromycin; Insulin Human Regular; NS; Ipratropium Bromide MDI; Methylprednisolone Na Succ.; Syringe (IV Room); Metolazone; Alteplase (Catheter Clearance); Oxymetazoline HCl; traZODONE HCl; Zolpidem Tartrate\n- Procedures performed: Temporary tracheostomy; Percutaneous [endoscopic] gastrostomy [PEG]; Other intubation of respiratory tract; Non-invasive mechanical ventilation; Insertion of endotracheal tube; Continuous invasive mechanical ventilation for 96 consecutive hours or more; Injection into thoracic cavity; Injection of antibiotic; Enteral infusion of concentrated nutritional substances; Venous catheterization, not elsewhere classified; Insertion of intercostal catheter for drainage; Transfusion of packed cells; Closed [endoscopic] biopsy of bronchus"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_005 — pair_id 410

- **Tags:** cross_model_disagreement, cross_model_disagreement_high, gemma_parse_salvage, gemma_partial_output, gpt_parse_salvage, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (4):** lipid, diabetes, hypertension, cardio
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 0, 'obesity': 0, 'cardio': 0, 'kidney': 0, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_salvaged_full` (salvage=True, n_tasks_salvaged=7)
- Wrong vs gold: ['lipid', 'diabetes', 'hypertension', 'cardio', 'kidney', 'stroke']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.8}, 'diabetes': {'pred': 1, 'prob': 0.9}, 'hypertension': {'pred': 1, 'prob': 0.85}, 'obesity': {'pred': 0, 'prob': 0.1}, 'cardio': {'pred': 1, 'prob': 0.75}, 'kidney': {'pred': 1, 'prob': 0.7}, 'stroke': {'pred': 1, 'prob': 0.6}}`

**Reasoning:**

```
nan
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: []
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 0389;5070;5990;00845;42731;2762;2449;4168;V1259. Current-visit narrative (truncated): - Diagnoses made: Unspecified septicemia; Pneumonitis due to inhalation of food or vomitus; Urinary tract infection, site not specified; Intestinal infection due to Clostridium difficile; Atrial fibrillation; Acidosis; Unspecified acquired hypothyroidism; Other chronic pulmonary heart diseases; Personal history of other diseases of circulatory system - Medications prescribed: NS; Norepinephrine; D5W; Iso-Osmotic Dextrose; SW; Potassium Chloride; Magnesium Sulfate; Cosyntropin; Ceftazidime; Vancomycin HCl; Dobutamine; Humulin-R Insulin; Aspirin; Pantoprazole Sodium; Metronidazole; Heparin; Potassium Phosphate; Levofloxacin; Dextrose 5%; Docusate Sodium; Atorvastatin; Senna; Bisacodyl; Levothyroxine Sodium; Megestrol Acetate; Calcium Carbonate; Magnesium Oxide; Neutra-Phos; Captopril; Metoprolol; Pantoprazole; Famotidine; Docusate Sodium (Liquid); Acetaminophen - Procedures performed: V...

### Current-visit narrative

```
- Diagnoses made: Unspecified septicemia; Pneumonitis due to inhalation of food or vomitus; Urinary tract infection, site not specified; Intestinal infection due to Clostridium difficile; Atrial fibrillation; Acidosis; Unspecified acquired hypothyroidism; Other chronic pulmonary heart diseases; Personal history of other diseases of circulatory system
- Medications prescribed: NS; Norepinephrine; D5W; Iso-Osmotic Dextrose; SW; Potassium Chloride; Magnesium Sulfate; Cosyntropin; Ceftazidime; Vancomycin HCl; Dobutamine; Humulin-R Insulin; Aspirin; Pantoprazole Sodium; Metronidazole; Heparin; Potassium Phosphate; Levofloxacin; Dextrose 5%; Docusate Sodium; Atorvastatin; Senna; Bisacodyl; Levothyroxine Sodium; Megestrol Acetate; Calcium Carbonate; Magnesium Oxide; Neutra-Phos; Captopril; Metoprolol; Pantoprazole; Famotidine; Docusate Sodium (Liquid); Acetaminophen
- Procedures performed: Venous catheterization, not elsewhere classified
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 410,
  "diagnoses_codes_current": "0389;5070;5990;00845;42731;2762;2449;4168;V1259",
  "procedures_codes_current": "3893",
  "medications_current": "NS;Norepinephrine;D5W;Iso-Osmotic Dextrose;SW;Potassium Chloride;Magnesium Sulfate;Cosyntropin;Ceftazidime;Vancomycin HCl;Dobutamine;Humulin-R Insulin;Aspirin;Pantoprazole Sodium;Metronidazole;Heparin;Potassium Phosphate;Levofloxacin;Dextrose 5%;Docusate Sodium;Atorvastatin;Senna;Bisacodyl;Levothyroxine Sodium;Megestrol Acetate;Calcium Carbonate;Magnesium Oxide;Neutra-Phos;Captopril;Metoprolol;Pantoprazole;Famotidine;Docusate Sodium (Liquid);Acetaminophen",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 0,
  "label_kidney_next": 0,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Unspecified septicemia; Pneumonitis due to inhalation of food or vomitus; Urinary tract infection, site not specified; Intestinal infection due to Clostridium difficile; Atrial fibrillation; Acidosis; Unspecified acquired hypothyroidism; Other chronic pulmonary heart diseases; Personal history of other diseases of circulatory system\n- Medications prescribed: NS; Norepinephrine; D5W; Iso-Osmotic Dextrose; SW; Potassium Chloride; Magnesium Sulfate; Cosyntropin; Ceftazidime; Vancomycin HCl; Dobutamine; Humulin-R Insulin; Aspirin; Pantoprazole Sodium; Metronidazole; Heparin; Potassium Phosphate; Levofloxacin; Dextrose 5%; Docusate Sodium; Atorvastatin; Senna; Bisacodyl; Levothyroxine Sodium; Megestrol Acetate; Calcium Carbonate; Magnesium Oxide; Neutra-Phos; Captopril; Metoprolol; Pantoprazole; Famotidine; Docusate Sodium (Liquid); Acetaminophen\n- Procedures performed: Venous catheterization, not elsewhere classified"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_006 — pair_id 507

- **Tags:** cross_model_disagreement, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (2):** lipid, diabetes
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 1, 'obesity': 0, 'cardio': 0, 'kidney': 0, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['diabetes', 'cardio', 'stroke']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 1, 'prob': 0.8}, 'hypertension': {'pred': 1, 'prob': 0.8}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.8}, 'kidney': {'pred': 0, 'prob': 0.0}, 'stroke': {'pred': 1, 'prob': 0.7}}`

**Reasoning:**

```
The patient has a history of atrial fibrillation and cardiac arrest, indicating cardiovascular issues, and is currently on insulin, suggesting diabetes. Unspecified essential hypertension is documented, confirming hypertension. There is no evidence of obesity in the record. The presence of anoxic brain damage and other complications raises the likelihood of a stroke in the next visit.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['lipid', 'cardio']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.95}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 1, 'prob': 0.95}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.95}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 5128;4275;496;5990;3481;41519;42731;V4611;51883;04111;4019;2859. Current-visit narrative (truncated): - Diagnoses made: [ICD9-DX:5128]; Cardiac arrest; Chronic airway obstruction, not elsewhere classified; Urinary tract infection, site not specified; Anoxic brain damage; Other pulmonary embolism and infarction; Atrial fibrillation; Dependence on respirator, status; Chronic respiratory failure; Methicillin susceptible Staphylococcus aureus in conditions classified elsewhere and of unspecified site; Unspecified essential hypertension; Anemia, unspecified - Medications prescribed: D5W; Fluticasone Propionate 110mcg; Albuterol; Ciprofloxacin; Quetiapine Fumarate; Multivitamins; Fluoxetine HCl; Ranitidine; Senna; Albuterol-Ipratropium; Theophylline SR; Heparin; Propofol; PredniSONE; Insulin; Potassium Chloride; MetRONIDAZOLE (FLagyl); NS; Levofloxacin; Magnesium Sulfate; Dextrose 5%; Docusate Sodium (Liquid); Theophylline ER (Uniphyl); Heparin Sodium; Vancomycin HCl; Iso-Osmotic Dextrose; ...

### Current-visit narrative

```
- Diagnoses made: [ICD9-DX:5128]; Cardiac arrest; Chronic airway obstruction, not elsewhere classified; Urinary tract infection, site not specified; Anoxic brain damage; Other pulmonary embolism and infarction; Atrial fibrillation; Dependence on respirator, status; Chronic respiratory failure; Methicillin susceptible Staphylococcus aureus in conditions classified elsewhere and of unspecified site; Unspecified essential hypertension; Anemia, unspecified
- Medications prescribed: D5W; Fluticasone Propionate 110mcg; Albuterol; Ciprofloxacin; Quetiapine Fumarate; Multivitamins; Fluoxetine HCl; Ranitidine; Senna; Albuterol-Ipratropium; Theophylline SR; Heparin; Propofol; PredniSONE; Insulin; Potassium Chloride; MetRONIDAZOLE (FLagyl); NS; Levofloxacin; Magnesium Sulfate; Dextrose 5%; Docusate Sodium (Liquid); Theophylline ER (Uniphyl); Heparin Sodium; Vancomycin HCl; Iso-Osmotic Dextrose; Acular; Latanoprost 0.005% Ophth. Soln.; Sodium Chloride 0.9%  Flush; Ipratropium Bromide MDI; PrednisoLONE Acetate 1% Ophth. Susp.; Bisacodyl; Lactulose; D5 1/2NS; Ipratropium Bromide Neb; Albuterol 0.083% Neb Soln; Fleet Phospho-Soda; Magnesium Citrate; Mineral Oil; Piperacillin-Tazobactam Na; Alprazolam; Lorazepam; Morphine Sulfate; Enoxaparin Sodium; Warfarin; Theophylline (Oral Solution); Gentamicin; Heparin Flush PICC (100 units/ml); Isotonic Sodium Chloride; SW
- Procedures performed: Continuous invasive mechanical ventilation for 96 consecutive hours or more; Venous catheterization, not elsewhere classified
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 507,
  "diagnoses_codes_current": "5128;4275;496;5990;3481;41519;42731;V4611;51883;04111;4019;2859",
  "procedures_codes_current": "9672;3893",
  "medications_current": "D5W;Fluticasone Propionate 110mcg;Albuterol;Ciprofloxacin;Quetiapine Fumarate;Multivitamins;Fluoxetine HCl;Ranitidine;Senna;Albuterol-Ipratropium;Theophylline SR;Heparin;Propofol;PredniSONE;Insulin;Potassium Chloride;MetRONIDAZOLE (FLagyl);NS;Levofloxacin;Magnesium Sulfate;Dextrose 5%;Docusate Sodium (Liquid);Theophylline ER (Uniphyl);Heparin Sodium;Vancomycin HCl;Iso-Osmotic Dextrose;Acular;Latanoprost 0.005% Ophth. Soln.;Sodium Chloride 0.9%  Flush;Ipratropium Bromide MDI;PrednisoLONE Acetate 1% Ophth. Susp.;Bisacodyl;Lactulose;D5 1/2NS;Ipratropium Bromide Neb;Albuterol 0.083% Neb Soln;Fleet Phospho-Soda;Magnesium Citrate;Mineral Oil;Piperacillin-Tazobactam Na;Alprazolam;Lorazepam;Morphine Sulfate;Enoxaparin Sodium;Warfarin;Theophylline (Oral Solution);Gentamicin;Heparin Flush PICC (100 units/ml);Isotonic Sodium Chloride;SW",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 1,
  "label_obesity_current": 0,
  "label_cardio_next": 0,
  "label_kidney_next": 0,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: [ICD9-DX:5128]; Cardiac arrest; Chronic airway obstruction, not elsewhere classified; Urinary tract infection, site not specified; Anoxic brain damage; Other pulmonary embolism and infarction; Atrial fibrillation; Dependence on respirator, status; Chronic respiratory failure; Methicillin susceptible Staphylococcus aureus in conditions classified elsewhere and of unspecified site; Unspecified essential hypertension; Anemia, unspecified\n- Medications prescribed: D5W; Fluticasone Propionate 110mcg; Albuterol; Ciprofloxacin; Quetiapine Fumarate; Multivitamins; Fluoxetine HCl; Ranitidine; Senna; Albuterol-Ipratropium; Theophylline SR; Heparin; Propofol; PredniSONE; Insulin; Potassium Chloride; MetRONIDAZOLE (FLagyl); NS; Levofloxacin; Magnesium Sulfate; Dextrose 5%; Docusate Sodium (Liquid); Theophylline ER (Uniphyl); Heparin Sodium; Vancomycin HCl; Iso-Osmotic Dextrose; Acular; Latanoprost 0.005% Ophth. Soln.; Sodium Chloride 0.9%  Flush; Ipratropium Bromide MDI; PrednisoLONE Acetate 1% Ophth. Susp.; Bisacodyl; Lactulose; D5 1/2NS; Ipratropium Bromide Neb; Albuterol 0.083% Neb Soln; Fleet Phospho-Soda; Magnesium Citrate; Mineral Oil; Piperacillin-Tazobactam Na; Alprazolam; Lorazepam; Morphine Sulfate; Enoxaparin Sodium; Warfarin; Theophylline (Oral Solution); Gentamicin; Heparin Flush PICC (100 units/ml); Isotonic Sodium Chloride; SW\n- Procedures performed: Continuous invasive mechanical ventilation for 96 consecutive hours or more; Venous catheterization, not elsewhere classified"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_007 — pair_id 798

- **Tags:** cross_model_disagreement, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement
- **Disagreeing tasks (2):** diabetes, cardio
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 0, 'obesity': 0, 'cardio': 1, 'kidney': 1, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: []
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.8}, 'kidney': {'pred': 1, 'prob': 0.75}, 'stroke': {'pred': 0, 'prob': 0.0}}`

**Reasoning:**

```
The patient has diagnoses of diastolic heart failure and congestive heart failure, indicating a high likelihood of cardiovascular disease. Acute kidney failure is also documented, suggesting a strong possibility of kidney disease in the next visit. There is no evidence of diabetes, hypertension, obesity, or disorders of lipid metabolism in the current visit record.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['diabetes', 'cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 1, 'prob': 0.95}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 51882;32723;49320;2554;2532;42830;4280;5849;2535;2762;2760;7580;4168;78901;7806. Current-visit narrative (truncated): - Diagnoses made: Other pulmonary insufficiency, not elsewhere classified; Obstructive sleep apnea (adult)(pediatric); Chronic obstructive asthma, unspecified; [ICD9-DX:2554]; Panhypopituitarism; Diastolic heart failure, unspecified; Congestive heart failure, unspecified; Acute kidney failure, unspecified; Diabetes insipidus; Acidosis; Hyperosmolality and/or hypernatremia; Down's syndrome; Other chronic pulmonary heart diseases; Abdominal pain, right upper quadrant; [ICD9-DX:7806] - Medications prescribed: Ipratropium Bromide MDI; Prednisone; Albuterol; Fludrocortisone Acetate; Hydrocortisone Na Succ.; 1/2 NS; Desmopressin Acetate; Albuterol 0.083% Neb Soln; D5W; Valsartan; Ipratropium Bromide Neb; Desmopressin Nasal; Insulin; Dextrose 5%; NS; Midazolam HCl; Metronidazole; Levofloxacin; Fentanyl Citrate; Levothyroxine Sodium; Furosemide; Heparin; Sodium Chloride 0.9% Flush; Aspirin; R...

### Current-visit narrative

```
- Diagnoses made: Other pulmonary insufficiency, not elsewhere classified; Obstructive sleep apnea (adult)(pediatric); Chronic obstructive asthma, unspecified; [ICD9-DX:2554]; Panhypopituitarism; Diastolic heart failure, unspecified; Congestive heart failure, unspecified; Acute kidney failure, unspecified; Diabetes insipidus; Acidosis; Hyperosmolality and/or hypernatremia; Down's syndrome; Other chronic pulmonary heart diseases; Abdominal pain, right upper quadrant; [ICD9-DX:7806]
- Medications prescribed: Ipratropium Bromide MDI; Prednisone; Albuterol; Fludrocortisone Acetate; Hydrocortisone Na Succ.; 1/2 NS; Desmopressin Acetate; Albuterol 0.083% Neb Soln; D5W; Valsartan; Ipratropium Bromide Neb; Desmopressin Nasal; Insulin; Dextrose 5%; NS; Midazolam HCl; Metronidazole; Levofloxacin; Fentanyl Citrate; Levothyroxine Sodium; Furosemide; Heparin; Sodium Chloride 0.9%  Flush; Aspirin; Ranitidine; Potassium Phosphate; Potassium Chloride; Docusate Sodium; Senna; Lidocaine Jelly 2% (Urojet)
- Procedures performed: Continuous invasive mechanical ventilation for less than 96 consecutive hours; Insertion of endotracheal tube; Arterial catheterization; Venous catheterization, not elsewhere classified; Non-invasive mechanical ventilation
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 798,
  "diagnoses_codes_current": "51882;32723;49320;2554;2532;42830;4280;5849;2535;2762;2760;7580;4168;78901;7806",
  "procedures_codes_current": "9671;9604;3891;3893;9390",
  "medications_current": "Ipratropium Bromide MDI;Prednisone;Albuterol;Fludrocortisone Acetate;Hydrocortisone Na Succ.;1/2 NS;Desmopressin Acetate;Albuterol 0.083% Neb Soln;D5W;Valsartan;Ipratropium Bromide Neb;Desmopressin Nasal;Insulin;Dextrose 5%;NS;Midazolam HCl;Metronidazole;Levofloxacin;Fentanyl Citrate;Levothyroxine Sodium;Furosemide;Heparin;Sodium Chloride 0.9%  Flush;Aspirin;Ranitidine;Potassium Phosphate;Potassium Chloride;Docusate Sodium;Senna;Lidocaine Jelly 2% (Urojet)",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 1,
  "label_kidney_next": 1,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Other pulmonary insufficiency, not elsewhere classified; Obstructive sleep apnea (adult)(pediatric); Chronic obstructive asthma, unspecified; [ICD9-DX:2554]; Panhypopituitarism; Diastolic heart failure, unspecified; Congestive heart failure, unspecified; Acute kidney failure, unspecified; Diabetes insipidus; Acidosis; Hyperosmolality and/or hypernatremia; Down's syndrome; Other chronic pulmonary heart diseases; Abdominal pain, right upper quadrant; [ICD9-DX:7806]\n- Medications prescribed: Ipratropium Bromide MDI; Prednisone; Albuterol; Fludrocortisone Acetate; Hydrocortisone Na Succ.; 1/2 NS; Desmopressin Acetate; Albuterol 0.083% Neb Soln; D5W; Valsartan; Ipratropium Bromide Neb; Desmopressin Nasal; Insulin; Dextrose 5%; NS; Midazolam HCl; Metronidazole; Levofloxacin; Fentanyl Citrate; Levothyroxine Sodium; Furosemide; Heparin; Sodium Chloride 0.9%  Flush; Aspirin; Ranitidine; Potassium Phosphate; Potassium Chloride; Docusate Sodium; Senna; Lidocaine Jelly 2% (Urojet)\n- Procedures performed: Continuous invasive mechanical ventilation for less than 96 consecutive hours; Insertion of endotracheal tube; Arterial catheterization; Venous catheterization, not elsewhere classified; Non-invasive mechanical ventilation"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_008 — pair_id 933

- **Tags:** cross_model_disagreement, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (1):** lipid
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 1, 'hypertension': 1, 'obesity': 0, 'cardio': 0, 'kidney': 1, 'stroke': 1}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['stroke']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 1, 'prob': 1.0}, 'hypertension': {'pred': 1, 'prob': 1.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': 1, 'prob': 1.0}, 'stroke': {'pred': 0, 'prob': 0.0}}`

**Reasoning:**

```
The patient has a documented diagnosis of diabetes mellitus and unspecified essential hypertension. There is no evidence of obesity or disorders of lipid metabolism. The patient has acute kidney failure, indicating a current kidney issue. There is no indication of cardiovascular disease or stroke in the current visit record.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['lipid']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.0}, 'diabetes': {'pred': 1, 'prob': 0.0}, 'hypertension': {'pred': 1, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 0389;78552;5990;5849;27651;99592;25000;4019. Current-visit narrative (truncated): - Diagnoses made: Unspecified septicemia; Septic shock; Urinary tract infection, site not specified; Acute kidney failure, unspecified; Dehydration; Severe sepsis; Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; Unspecified essential hypertension - Medications prescribed: GlyBURIDE; Valsartan; Ciprofloxacin HCl; GlipiZIDE; Meperidine; DiphenhydrAMINE HCl; NS; Piperacillin-Tazobactam Na; Vancomycin HCl; Iso-Osmotic Dextrose; D5W; Norepinephrine; Sodium Chloride 0.9% Flush; Clopidogrel Bisulfate; Aspirin EC; Pravastatin; Insulin; Allopurinol; Heparin; Acetaminophen; Senna; Bisacodyl; Ipratropium Bromide Neb - Procedures performed: Arterial catheterization; Venous catheterization, not elsewhere classified

### Current-visit narrative

```
- Diagnoses made: Unspecified septicemia; Septic shock; Urinary tract infection, site not specified; Acute kidney failure, unspecified; Dehydration; Severe sepsis; Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; Unspecified essential hypertension
- Medications prescribed: GlyBURIDE; Valsartan; Ciprofloxacin HCl; GlipiZIDE; Meperidine; DiphenhydrAMINE HCl; NS; Piperacillin-Tazobactam Na; Vancomycin HCl; Iso-Osmotic Dextrose; D5W; Norepinephrine; Sodium Chloride 0.9%  Flush; Clopidogrel Bisulfate; Aspirin EC; Pravastatin; Insulin; Allopurinol; Heparin; Acetaminophen; Senna; Bisacodyl; Ipratropium Bromide Neb
- Procedures performed: Arterial catheterization; Venous catheterization, not elsewhere classified
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 933,
  "diagnoses_codes_current": "0389;78552;5990;5849;27651;99592;25000;4019",
  "procedures_codes_current": "3891;3893",
  "medications_current": "GlyBURIDE;Valsartan;Ciprofloxacin HCl;GlipiZIDE;Meperidine;DiphenhydrAMINE HCl;NS;Piperacillin-Tazobactam Na;Vancomycin HCl;Iso-Osmotic Dextrose;D5W;Norepinephrine;Sodium Chloride 0.9%  Flush;Clopidogrel Bisulfate;Aspirin EC;Pravastatin;Insulin;Allopurinol;Heparin;Acetaminophen;Senna;Bisacodyl;Ipratropium Bromide Neb",
  "label_lipid_next": 0,
  "label_diabetes_current": 1,
  "label_hypertension_current": 1,
  "label_obesity_current": 0,
  "label_cardio_next": 0,
  "label_kidney_next": 1,
  "label_stroke_next": 1,
  "narrative_current": "- Diagnoses made: Unspecified septicemia; Septic shock; Urinary tract infection, site not specified; Acute kidney failure, unspecified; Dehydration; Severe sepsis; Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; Unspecified essential hypertension\n- Medications prescribed: GlyBURIDE; Valsartan; Ciprofloxacin HCl; GlipiZIDE; Meperidine; DiphenhydrAMINE HCl; NS; Piperacillin-Tazobactam Na; Vancomycin HCl; Iso-Osmotic Dextrose; D5W; Norepinephrine; Sodium Chloride 0.9%  Flush; Clopidogrel Bisulfate; Aspirin EC; Pravastatin; Insulin; Allopurinol; Heparin; Acetaminophen; Senna; Bisacodyl; Ipratropium Bromide Neb\n- Procedures performed: Arterial catheterization; Venous catheterization, not elsewhere classified"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_009 — pair_id 956

- **Tags:** cross_model_disagreement, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (2):** lipid, diabetes
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 1, 'obesity': 1, 'cardio': 1, 'kidney': 1, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['diabetes']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 1, 'prob': 0.8}, 'hypertension': {'pred': 1, 'prob': 0.8}, 'obesity': {'pred': 1, 'prob': 0.9}, 'cardio': {'pred': 1, 'prob': 0.7}, 'kidney': {'pred': 1, 'prob': 0.8}, 'stroke': {'pred': 0, 'prob': 0.0}}`

**Reasoning:**

```
The patient has a history of morbid obesity, unspecified essential hypertension, and is currently receiving insulin, which suggests diabetes. There is also evidence of acute kidney failure and congestive heart failure, indicating a high likelihood of cardiovascular issues. However, there is no explicit mention of lipid disorders or stroke risk in the current visit record.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['lipid']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.95}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 1, 'prob': 0.95}, 'obesity': {'pred': 1, 'prob': 0.95}, 'cardio': {'pred': 1, 'prob': 0.95}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 7837;4280;51881;5070;0389;99592;5849;27801;32723;4019;V1588;29680;1103;319;2859. Current-visit narrative (truncated): - Diagnoses made: Adult failure to thrive; Congestive heart failure, unspecified; Acute respiratory failure; Pneumonitis due to inhalation of food or vomitus; Unspecified septicemia; Severe sepsis; Acute kidney failure, unspecified; Morbid obesity; Obstructive sleep apnea (adult)(pediatric); Unspecified essential hypertension; History of fall; Bipolar disorder, unspecified; Dermatophytosis of groin and perianal area; Unspecified intellectual disabilities; Anemia, unspecified - Medications prescribed: Aspirin EC; Fentanyl Citrate; Levofloxacin; Vancomycin HCl; D5W; NS; Dextrose 5%; Furosemide; Captopril; Aspirin; Docusate Sodium (Liquid); Senna; Bisacodyl; CeftAZIDime; Iso-Osmotic Dextrose; Ipratropium Bromide Neb; Albuterol 0.083% Neb Soln; Quetiapine Fumarate; Acetaminophen; Ibuprofen; Docusate Sodium; Ipratropium Bromide MDI; Albuterol; Pantoprazole; Fluticasone Propionate 110mcg; L...

### Current-visit narrative

```
- Diagnoses made: Adult failure to thrive; Congestive heart failure, unspecified; Acute respiratory failure; Pneumonitis due to inhalation of food or vomitus; Unspecified septicemia; Severe sepsis; Acute kidney failure, unspecified; Morbid obesity; Obstructive sleep apnea (adult)(pediatric); Unspecified essential hypertension; History of fall; Bipolar disorder, unspecified; Dermatophytosis of groin and perianal area; Unspecified intellectual disabilities; Anemia, unspecified
- Medications prescribed: Aspirin EC; Fentanyl Citrate; Levofloxacin; Vancomycin HCl; D5W; NS; Dextrose 5%; Furosemide; Captopril; Aspirin; Docusate Sodium (Liquid); Senna; Bisacodyl; CeftAZIDime; Iso-Osmotic Dextrose; Ipratropium Bromide Neb; Albuterol 0.083% Neb Soln; Quetiapine Fumarate; Acetaminophen; Ibuprofen; Docusate Sodium; Ipratropium Bromide MDI; Albuterol; Pantoprazole; Fluticasone Propionate 110mcg; Lisinopril; Insulin; Montelukast Sodium; Heparin; Miconazole Powder 2%; Lithium Carbonate; Cosyntropin; Midazolam HCl; Metronidazole; Pantoprazole Sodium; Vial; Artificial Tears; Chlorhexidine Gluconate; Heparin Flush CVL  (100 units/ml); Norepinephrine; Ibuprofen Suspension; Acetaminophen (Liquid)
- Procedures performed: Insertion of endotracheal tube; Venous catheterization, not elsewhere classified; Non-invasive mechanical ventilation; Continuous invasive mechanical ventilation for 96 consecutive hours or more; Enteral infusion of concentrated nutritional substances; Infusion of vasopressor agent
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 956,
  "diagnoses_codes_current": "7837;4280;51881;5070;0389;99592;5849;27801;32723;4019;V1588;29680;1103;319;2859",
  "procedures_codes_current": "9604;3893;9390;9672;966;17",
  "medications_current": "Aspirin EC;Fentanyl Citrate;Levofloxacin;Vancomycin HCl;D5W;NS;Dextrose 5%;Furosemide;Captopril;Aspirin;Docusate Sodium (Liquid);Senna;Bisacodyl;CeftAZIDime;Iso-Osmotic Dextrose;Ipratropium Bromide Neb;Albuterol 0.083% Neb Soln;Quetiapine Fumarate;Acetaminophen;Ibuprofen;Docusate Sodium;Ipratropium Bromide MDI;Albuterol;Pantoprazole;Fluticasone Propionate 110mcg;Lisinopril;Insulin;Montelukast Sodium;Heparin;Miconazole Powder 2%;Lithium Carbonate;Cosyntropin;Midazolam HCl;Metronidazole;Pantoprazole Sodium;Vial;Artificial Tears;Chlorhexidine Gluconate;Heparin Flush CVL  (100 units/ml);Norepinephrine;Ibuprofen Suspension;Acetaminophen (Liquid)",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 1,
  "label_obesity_current": 1,
  "label_cardio_next": 1,
  "label_kidney_next": 1,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Adult failure to thrive; Congestive heart failure, unspecified; Acute respiratory failure; Pneumonitis due to inhalation of food or vomitus; Unspecified septicemia; Severe sepsis; Acute kidney failure, unspecified; Morbid obesity; Obstructive sleep apnea (adult)(pediatric); Unspecified essential hypertension; History of fall; Bipolar disorder, unspecified; Dermatophytosis of groin and perianal area; Unspecified intellectual disabilities; Anemia, unspecified\n- Medications prescribed: Aspirin EC; Fentanyl Citrate; Levofloxacin; Vancomycin HCl; D5W; NS; Dextrose 5%; Furosemide; Captopril; Aspirin; Docusate Sodium (Liquid); Senna; Bisacodyl; CeftAZIDime; Iso-Osmotic Dextrose; Ipratropium Bromide Neb; Albuterol 0.083% Neb Soln; Quetiapine Fumarate; Acetaminophen; Ibuprofen; Docusate Sodium; Ipratropium Bromide MDI; Albuterol; Pantoprazole; Fluticasone Propionate 110mcg; Lisinopril; Insulin; Montelukast Sodium; Heparin; Miconazole Powder 2%; Lithium Carbonate; Cosyntropin; Midazolam HCl; Metronidazole; Pantoprazole Sodium; Vial; Artificial Tears; Chlorhexidine Gluconate; Heparin Flush CVL  (100 units/ml); Norepinephrine; Ibuprofen Suspension; Acetaminophen (Liquid)\n- Procedures performed: Insertion of endotracheal tube; Venous catheterization, not elsewhere classified; Non-invasive mechanical ventilation; Continuous invasive mechanical ventilation for 96 consecutive hours or more; Enteral infusion of concentrated nutritional substances; Infusion of vasopressor agent"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_010 — pair_id 1505

- **Tags:** cross_model_disagreement, cross_model_disagreement_high, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (4):** lipid, diabetes, hypertension, cardio
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 0, 'obesity': 0, 'cardio': 0, 'kidney': 0, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['lipid', 'diabetes', 'hypertension', 'cardio', 'kidney']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.7}, 'diabetes': {'pred': 1, 'prob': 0.6}, 'hypertension': {'pred': 1, 'prob': 0.65}, 'obesity': {'pred': 0, 'prob': 0.1}, 'cardio': {'pred': 1, 'prob': 0.8}, 'kidney': {'pred': 1, 'prob': 0.75}, 'stroke': {'pred': 0, 'prob': 0.2}}`

**Reasoning:**

```
The patient is prescribed Atorvastatin, which indicates a history of lipid disorders. There is also evidence of diabetes management with Insulin. The presence of congestive heart failure and other vascular issues suggests hypertension. The patient's weight status is not documented, leading to a low probability of obesity. Given the cardiovascular complications, the likelihood of future cardiovascular disease is high. The patient's current condition and procedures suggest a risk for kidney disease. There is no direct evidence of stroke risk in the current visit.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: []
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 5571;5185;99674;56881;99811;9973;486;4280;4928;99859;9982;2859. Current-visit narrative (truncated): - Diagnoses made: Chronic vascular insufficiency of intestine; [ICD9-DX:5185]; Other complications due to other vascular device, implant, and graft; Hemoperitoneum (nontraumatic); Hemorrhage complicating a procedure; [ICD9-DX:9973]; Pneumonia, organism unspecified; Congestive heart failure, unspecified; Other emphysema; Other postoperative infection; Accidental puncture or laceration during a procedure, not elsewhere classified; Anemia, unspecified - Medications prescribed: SW; Hydromorphone; Furosemide; Fluconazole; Metronidazole; Levofloxacin; Potassium Chloride; Magnesium Sulfate; Morphine Sulfate; Norepinephrine; D5W; NS; Vasopressin; Vancomycin HCl; Iso-Osmotic Dextrose; Albumin 5% (12.5 g); Insulin; Fondaparinux; Metoprolol; Acetaminophen; Lorazepam; Vial; Pantoprazole Sodium; Dextrose 5%; LR; Etomidate; Succinylcholine; Midazolam HCl; Propofol; Fentanyl Citrate; Calcium Glucona...

### Current-visit narrative

```
- Diagnoses made: Chronic vascular insufficiency of intestine; [ICD9-DX:5185]; Other complications due to other vascular device, implant, and graft; Hemoperitoneum (nontraumatic); Hemorrhage complicating a procedure; [ICD9-DX:9973]; Pneumonia, organism unspecified; Congestive heart failure, unspecified; Other emphysema; Other postoperative infection; Accidental puncture or laceration during a procedure, not elsewhere classified; Anemia, unspecified
- Medications prescribed: SW; Hydromorphone; Furosemide; Fluconazole; Metronidazole; Levofloxacin; Potassium Chloride; Magnesium Sulfate; Morphine Sulfate; Norepinephrine; D5W; NS; Vasopressin; Vancomycin HCl; Iso-Osmotic Dextrose; Albumin 5% (12.5 g); Insulin; Fondaparinux; Metoprolol; Acetaminophen; Lorazepam; Vial; Pantoprazole Sodium; Dextrose 5%; LR; Etomidate; Succinylcholine; Midazolam HCl; Propofol; Fentanyl Citrate; Calcium Gluconate; Nitroglycerin; Amino Acids 4.25% W/ Dextrose 5%; Multivitamin IV (Without Vit. K); AcetaZOLAMIDE; AcetaZOLAMIDE Sodium; D10W; DiphenhydrAMINE HCl; Potassium Chl 20 mEq / 1000 mL D5 1/2 NS; Lidocaine Jelly 2% (Urojet); Aspirin; Clopidogrel Bisulfate; Atorvastatin; Folic Acid; Albuterol-Ipratropium; Albuterol; Iron Dextran Complex; Caspofungin; Dolasetron Mesylate; Albuterol 0.083% Neb Soln
- Procedures performed: Other intra-abdominal vascular shunt or bypass; Other partial resection of small intestine; Other revision of vascular procedure; Suture of vein; Reopening of recent laparotomy site; Total removal of small intestine; Open and other right hemicolectomy; Temporary colostomy; Other gastrostomy; Other enterostomy; Closed [endoscopic] biopsy of bronchus; Continuous invasive mechanical ventilation for 96 consecutive hours or more; Parenteral infusion of concentrated nutritional substances; Venous catheterization, not elsewhere classified
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 1505,
  "diagnoses_codes_current": "5571;5185;99674;56881;99811;9973;486;4280;4928;99859;9982;2859",
  "procedures_codes_current": "3926;4562;3949;3932;5412;4563;4573;4611;4319;4639;3324;9672;9915;3893",
  "medications_current": "SW;Hydromorphone;Furosemide;Fluconazole;Metronidazole;Levofloxacin;Potassium Chloride;Magnesium Sulfate;Morphine Sulfate;Norepinephrine;D5W;NS;Vasopressin;Vancomycin HCl;Iso-Osmotic Dextrose;Albumin 5% (12.5 g);Insulin;Fondaparinux;Metoprolol;Acetaminophen;Lorazepam;Vial;Pantoprazole Sodium;Dextrose 5%;LR;Etomidate;Succinylcholine;Midazolam HCl;Propofol;Fentanyl Citrate;Calcium Gluconate;Nitroglycerin;Amino Acids 4.25% W/ Dextrose 5%;Multivitamin IV (Without Vit. K);AcetaZOLAMIDE;AcetaZOLAMIDE Sodium;D10W;DiphenhydrAMINE HCl;Potassium Chl 20 mEq / 1000 mL D5 1/2 NS;Lidocaine Jelly 2% (Urojet);Aspirin;Clopidogrel Bisulfate;Atorvastatin;Folic Acid;Albuterol-Ipratropium;Albuterol;Iron Dextran Complex;Caspofungin;Dolasetron Mesylate;Albuterol 0.083% Neb Soln;Ipratropium Bromide Neb;Ferrous Sulfate;Sodium Chloride 0.9%  Flush;Albumin 5% (25 g);Phytonadione;Potassium Phosphate;Lepirudin;D5 1/2NS;Vitamin B Complex w/C;Sucralfate;Famotidine;Nitroglycerin SL;Aspirin EC;Phenylephrine HCl",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 0,
  "label_kidney_next": 0,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Chronic vascular insufficiency of intestine; [ICD9-DX:5185]; Other complications due to other vascular device, implant, and graft; Hemoperitoneum (nontraumatic); Hemorrhage complicating a procedure; [ICD9-DX:9973]; Pneumonia, organism unspecified; Congestive heart failure, unspecified; Other emphysema; Other postoperative infection; Accidental puncture or laceration during a procedure, not elsewhere classified; Anemia, unspecified\n- Medications prescribed: SW; Hydromorphone; Furosemide; Fluconazole; Metronidazole; Levofloxacin; Potassium Chloride; Magnesium Sulfate; Morphine Sulfate; Norepinephrine; D5W; NS; Vasopressin; Vancomycin HCl; Iso-Osmotic Dextrose; Albumin 5% (12.5 g); Insulin; Fondaparinux; Metoprolol; Acetaminophen; Lorazepam; Vial; Pantoprazole Sodium; Dextrose 5%; LR; Etomidate; Succinylcholine; Midazolam HCl; Propofol; Fentanyl Citrate; Calcium Gluconate; Nitroglycerin; Amino Acids 4.25% W/ Dextrose 5%; Multivitamin IV (Without Vit. K); AcetaZOLAMIDE; AcetaZOLAMIDE Sodium; D10W; DiphenhydrAMINE HCl; Potassium Chl 20 mEq / 1000 mL D5 1/2 NS; Lidocaine Jelly 2% (Urojet); Aspirin; Clopidogrel Bisulfate; Atorvastatin; Folic Acid; Albuterol-Ipratropium; Albuterol; Iron Dextran Complex; Caspofungin; Dolasetron Mesylate; Albuterol 0.083% Neb Soln\n- Procedures performed: Other intra-abdominal vascular shunt or bypass; Other partial resection of small intestine; Other revision of vascular procedure; Suture of vein; Reopening of recent laparotomy site; Total removal of small intestine; Open and other right hemicolectomy; Temporary colostomy; Other gastrostomy; Other enterostomy; Closed [endoscopic] biopsy of bronchus; Continuous invasive mechanical ventilation for 96 consecutive hours or more; Parenteral infusion of concentrated nutritional substances; Venous catheterization, not elsewhere classified"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_011 — pair_id 2347

- **Tags:** cross_model_disagreement, cross_model_disagreement_high, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (4):** lipid, diabetes, hypertension, cardio
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 0, 'obesity': 1, 'cardio': 1, 'kidney': 1, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['lipid', 'diabetes', 'hypertension']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.8}, 'diabetes': {'pred': 1, 'prob': 0.7}, 'hypertension': {'pred': 1, 'prob': 0.7}, 'obesity': {'pred': 1, 'prob': 0.9}, 'cardio': {'pred': 1, 'prob': 0.85}, 'kidney': {'pred': 1, 'prob': 0.8}, 'stroke': {'pred': 0, 'prob': 0.1}}`

**Reasoning:**

```
The patient has a history of coronary atherosclerosis and is prescribed Simvastatin, indicating a lipid disorder. The presence of insulin suggests diabetes, and Lisinopril indicates hypertension. Obesity is explicitly mentioned. The patient has acute and chronic kidney failure, suggesting ongoing kidney issues. Cardiovascular disease is likely due to existing heart conditions. There is no evidence of stroke risk in the current record.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 1, 'prob': 0.8}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 42833;51884;42732;7455;5849;2760;2763;4280;V440;71590;4240;2809;5859;32723;27800;2875;41401;2749;V103. Current-visit narrative (truncated): - Diagnoses made: Acute on chronic diastolic heart failure; Acute and chronic respiratory failure; Atrial flutter; Ostium secundum type atrial septal defect; Acute kidney failure, unspecified; Hyperosmolality and/or hypernatremia; Alkalosis; Congestive heart failure, unspecified; Tracheostomy status; Osteoarthrosis, unspecified whether generalized or localized, site unspecified; Mitral valve disorders; Iron deficiency anemia, unspecified; Chronic kidney disease, unspecified; Obstructive sleep apnea (adult)(pediatric); Obesity, unspecified; Thrombocytopenia, unspecified; Coronary atherosclerosis of native coronary artery; Gout, unspecified; Personal history of malignant neoplasm of breast - Medications prescribed: Ipratropium Bromide Neb; Albuterol 0.083% Neb Soln; Heparin; Colchicine; Simvastatin; Ranitidine; Lisinopril; Ferrous Gluconate; Anastrozole; Calcitriol; Sodium Chloride 0.9%...

### Current-visit narrative

```
- Diagnoses made: Acute on chronic diastolic heart failure; Acute and chronic respiratory failure; Atrial flutter; Ostium secundum type atrial septal defect; Acute kidney failure, unspecified; Hyperosmolality and/or hypernatremia; Alkalosis; Congestive heart failure, unspecified; Tracheostomy status; Osteoarthrosis, unspecified whether generalized or localized, site unspecified; Mitral valve disorders; Iron deficiency anemia, unspecified; Chronic kidney disease, unspecified; Obstructive sleep apnea (adult)(pediatric); Obesity, unspecified; Thrombocytopenia, unspecified; Coronary atherosclerosis of native coronary artery; Gout, unspecified; Personal history of malignant neoplasm of breast
- Medications prescribed: Ipratropium Bromide Neb; Albuterol 0.083% Neb Soln; Heparin; Colchicine; Simvastatin; Ranitidine; Lisinopril; Ferrous Gluconate; Anastrozole; Calcitriol; Sodium Chloride 0.9%  Flush; Insulin; Furosemide; SW; Magnesium Sulfate; 5% Dextrose; AcetaZOLamide Sodium; Warfarin; *NF* Arginine HCl; Soln; 0.9% Sodium Chloride; Potassium Chloride
- Procedures performed: None recorded
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 2347,
  "diagnoses_codes_current": "42833;51884;42732;7455;5849;2760;2763;4280;V440;71590;4240;2809;5859;32723;27800;2875;41401;2749;V103",
  "procedures_codes_current": NaN,
  "medications_current": "Ipratropium Bromide Neb;Albuterol 0.083% Neb Soln;Heparin;Colchicine;Simvastatin;Ranitidine;Lisinopril;Ferrous Gluconate;Anastrozole;Calcitriol;Sodium Chloride 0.9%  Flush;Insulin;Furosemide;SW;Magnesium Sulfate;5% Dextrose;AcetaZOLamide Sodium;Warfarin;*NF* Arginine HCl;Soln;0.9% Sodium Chloride;Potassium Chloride",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 1,
  "label_cardio_next": 1,
  "label_kidney_next": 1,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Acute on chronic diastolic heart failure; Acute and chronic respiratory failure; Atrial flutter; Ostium secundum type atrial septal defect; Acute kidney failure, unspecified; Hyperosmolality and/or hypernatremia; Alkalosis; Congestive heart failure, unspecified; Tracheostomy status; Osteoarthrosis, unspecified whether generalized or localized, site unspecified; Mitral valve disorders; Iron deficiency anemia, unspecified; Chronic kidney disease, unspecified; Obstructive sleep apnea (adult)(pediatric); Obesity, unspecified; Thrombocytopenia, unspecified; Coronary atherosclerosis of native coronary artery; Gout, unspecified; Personal history of malignant neoplasm of breast\n- Medications prescribed: Ipratropium Bromide Neb; Albuterol 0.083% Neb Soln; Heparin; Colchicine; Simvastatin; Ranitidine; Lisinopril; Ferrous Gluconate; Anastrozole; Calcitriol; Sodium Chloride 0.9%  Flush; Insulin; Furosemide; SW; Magnesium Sulfate; 5% Dextrose; AcetaZOLamide Sodium; Warfarin; *NF* Arginine HCl; Soln; 0.9% Sodium Chloride; Potassium Chloride\n- Procedures performed: None recorded"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_012 — pair_id 2361

- **Tags:** cross_model_disagreement, cross_model_disagreement_high, gemma_parse_salvage, gemma_partial_output, gpt_parse_salvage, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (4):** lipid, diabetes, hypertension, cardio
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 0, 'obesity': 0, 'cardio': 0, 'kidney': 1, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_salvaged_full` (salvage=True, n_tasks_salvaged=7)
- Wrong vs gold: ['lipid', 'diabetes', 'hypertension', 'cardio']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.8}, 'diabetes': {'pred': 1, 'prob': 0.7}, 'hypertension': {'pred': 1, 'prob': 0.75}, 'obesity': {'pred': 0, 'prob': 0.1}, 'cardio': {'pred': 1, 'prob': 0.85}, 'kidney': {'pred': 1, 'prob': 0.9}, 'stroke': {'pred': 0, 'prob': 0.2}}`

**Reasoning:**

```
nan
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: []
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 5845;27651;51881;5070;4280;48283;42731;2851;34982;2764;3241;4589;3320;73018. Current-visit narrative (truncated): - Diagnoses made: Acute kidney failure with lesion of tubular necrosis; Dehydration; Acute respiratory failure; Pneumonitis due to inhalation of food or vomitus; Congestive heart failure, unspecified; Pneumonia due to other gram-negative bacteria; Atrial fibrillation; Acute posthemorrhagic anemia; Toxic encephalopathy; Mixed acid-base balance disorder; Intraspinal abscess; Hypotension, unspecified; Paralysis agitans; Chronic osteomyelitis, other specified sites - Medications prescribed: Magnesium Sulfate; Diltiazem; Levofloxacin; Sodium Chloride 0.9% Flush; D5 1/2NS; Vancomycin HCl; Iso-Osmotic Dextrose; Potassium Chloride; Warfarin; Metronidazole; Carbidopa-Levodopa (25-100); Calcium Carbonate; 1/2 NS; D5W; Albuterol 0.083% Neb Soln; Furosemide; Metoprolol; Lansoprazole; Carbidopa-Levodopa CR (25-100); Aspirin; Atorvastatin; NS; Carbidopa-Levodopa (25-250); Senna; Docusate Sodium (Li...

### Current-visit narrative

```
- Diagnoses made: Acute kidney failure with lesion of tubular necrosis; Dehydration; Acute respiratory failure; Pneumonitis due to inhalation of food or vomitus; Congestive heart failure, unspecified; Pneumonia due to other gram-negative bacteria; Atrial fibrillation; Acute posthemorrhagic anemia; Toxic encephalopathy; Mixed acid-base balance disorder; Intraspinal abscess; Hypotension, unspecified; Paralysis agitans; Chronic osteomyelitis, other specified sites
- Medications prescribed: Magnesium Sulfate; Diltiazem; Levofloxacin; Sodium Chloride 0.9%  Flush; D5 1/2NS; Vancomycin HCl; Iso-Osmotic Dextrose; Potassium Chloride; Warfarin; Metronidazole; Carbidopa-Levodopa (25-100); Calcium Carbonate; 1/2 NS; D5W; Albuterol 0.083% Neb Soln; Furosemide; Metoprolol; Lansoprazole; Carbidopa-Levodopa CR (25-100); Aspirin; Atorvastatin; NS; Carbidopa-Levodopa (25-250); Senna; Docusate Sodium (Liquid); Pantoprazole Sodium; Vial; Acetaminophen; Lorazepam; LR; Dextrose 5%; NS (Mini Bag Plus); Imipenem-Cilastatin; Insulin; Bisacodyl; Unasyn; Captopril; Thiamine HCl; Cyanocobalamin; Multivitamins; Digoxin; Pantoprazole; Heparin Flush CVL  (100 units/ml); Heparin; Propofol; Cosyntropin; Midazolam HCl; Fentanyl Citrate; Norepinephrine; Tobramycin Inhalation Soln
- Procedures performed: Insertion of endotracheal tube; Infusion of vasopressor agent; Closed [endoscopic] biopsy of bronchus; Transfusion of packed cells; Enteral infusion of concentrated nutritional substances; Arterial catheterization
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 2361,
  "diagnoses_codes_current": "5845;27651;51881;5070;4280;48283;42731;2851;34982;2764;3241;4589;3320;73018",
  "procedures_codes_current": "9604;17;3324;9904;966;3891",
  "medications_current": "Magnesium Sulfate;Diltiazem;Levofloxacin;Sodium Chloride 0.9%  Flush;D5 1/2NS;Vancomycin HCl;Iso-Osmotic Dextrose;Potassium Chloride;Warfarin;Metronidazole;Carbidopa-Levodopa (25-100);Calcium Carbonate;1/2 NS;D5W;Albuterol 0.083% Neb Soln;Furosemide;Metoprolol;Lansoprazole;Carbidopa-Levodopa CR (25-100);Aspirin;Atorvastatin;NS;Carbidopa-Levodopa (25-250);Senna;Docusate Sodium (Liquid);Pantoprazole Sodium;Vial;Acetaminophen;Lorazepam;LR;Dextrose 5%;NS (Mini Bag Plus);Imipenem-Cilastatin;Insulin;Bisacodyl;Unasyn;Captopril;Thiamine HCl;Cyanocobalamin;Multivitamins;Digoxin;Pantoprazole;Heparin Flush CVL  (100 units/ml);Heparin;Propofol;Cosyntropin;Midazolam HCl;Fentanyl Citrate;Norepinephrine;Tobramycin Inhalation Soln;Guaifenesin",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 0,
  "label_kidney_next": 1,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Acute kidney failure with lesion of tubular necrosis; Dehydration; Acute respiratory failure; Pneumonitis due to inhalation of food or vomitus; Congestive heart failure, unspecified; Pneumonia due to other gram-negative bacteria; Atrial fibrillation; Acute posthemorrhagic anemia; Toxic encephalopathy; Mixed acid-base balance disorder; Intraspinal abscess; Hypotension, unspecified; Paralysis agitans; Chronic osteomyelitis, other specified sites\n- Medications prescribed: Magnesium Sulfate; Diltiazem; Levofloxacin; Sodium Chloride 0.9%  Flush; D5 1/2NS; Vancomycin HCl; Iso-Osmotic Dextrose; Potassium Chloride; Warfarin; Metronidazole; Carbidopa-Levodopa (25-100); Calcium Carbonate; 1/2 NS; D5W; Albuterol 0.083% Neb Soln; Furosemide; Metoprolol; Lansoprazole; Carbidopa-Levodopa CR (25-100); Aspirin; Atorvastatin; NS; Carbidopa-Levodopa (25-250); Senna; Docusate Sodium (Liquid); Pantoprazole Sodium; Vial; Acetaminophen; Lorazepam; LR; Dextrose 5%; NS (Mini Bag Plus); Imipenem-Cilastatin; Insulin; Bisacodyl; Unasyn; Captopril; Thiamine HCl; Cyanocobalamin; Multivitamins; Digoxin; Pantoprazole; Heparin Flush CVL  (100 units/ml); Heparin; Propofol; Cosyntropin; Midazolam HCl; Fentanyl Citrate; Norepinephrine; Tobramycin Inhalation Soln\n- Procedures performed: Insertion of endotracheal tube; Infusion of vasopressor agent; Closed [endoscopic] biopsy of bronchus; Transfusion of packed cells; Enteral infusion of concentrated nutritional substances; Arterial catheterization"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_013 — pair_id 3839

- **Tags:** cross_model_disagreement, cross_model_disagreement_high, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (4):** lipid, diabetes, hypertension, cardio
- **Ground truth (task → label):** `{'lipid': 1, 'diabetes': 0, 'hypertension': 0, 'obesity': 0, 'cardio': 1, 'kidney': 1, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['diabetes', 'hypertension', 'kidney']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.8}, 'diabetes': {'pred': 1, 'prob': 0.9}, 'hypertension': {'pred': 1, 'prob': 0.85}, 'obesity': {'pred': 0, 'prob': 0.1}, 'cardio': {'pred': 1, 'prob': 0.75}, 'kidney': {'pred': 0, 'prob': 0.2}, 'stroke': {'pred': 0, 'prob': 0.15}}`

**Reasoning:**

```
The patient is prescribed Atorvastatin, which is used for lipid disorders, indicating a likelihood of a lipid metabolism disorder. Insulin is prescribed, suggesting current diabetes. Medications like Metoprolol and Hydralazine indicate hypertension management. There is no evidence of obesity in the record. The presence of acute venous embolism suggests a risk for cardiovascular issues. There is no indication of kidney disease or stroke risk based on the current visit record.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['lipid', 'cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 5191;5185;496;2762;45340;5533. Current-visit narrative (truncated): - Diagnoses made: [ICD9-DX:5191]; [ICD9-DX:5185]; Chronic airway obstruction, not elsewhere classified; Acidosis; Acute venous embolism and thrombosis of unspecified deep vessels of lower extremity; Diaphragmatic hernia without mention of obstruction or gangrene - Medications prescribed: Vial; D5W; Pantoprazole Sodium; Levothyroxine Sodium; Heparin Flush PICC (100 units/ml); Acetaminophen; Heparin Sodium; Sodium Chloride Nasal; Sarna Lotion; Insulin; Multivitamin IV; Amino Acids 4.25% W/ Dextrose 5%; Enoxaparin Sodium; Nystatin Cream; Linezolid; Atorvastatin; Escitalopram Oxalate; Pantoprazole; traZODONE HCl; Tolterodine; Sodium Chloride 0.9% Flush; Potassium Chloride; Furosemide; Vancomycin HCl; Fentanyl Citrate; Insulin Human Regular; NS; Zolpidem Tartrate; Acetylcysteine 20%; Fluticasone Propionate 110mcg; Hydralazine HCl; Lansoprazole Oral Suspension; Metoprolol; Acetaminophen (Li...

### Current-visit narrative

```
- Diagnoses made: [ICD9-DX:5191]; [ICD9-DX:5185]; Chronic airway obstruction, not elsewhere classified; Acidosis; Acute venous embolism and thrombosis of unspecified deep vessels of lower extremity; Diaphragmatic hernia without mention of obstruction or gangrene
- Medications prescribed: Vial; D5W; Pantoprazole Sodium; Levothyroxine Sodium; Heparin Flush PICC (100 units/ml); Acetaminophen; Heparin Sodium; Sodium Chloride Nasal; Sarna Lotion; Insulin; Multivitamin IV; Amino Acids 4.25% W/ Dextrose 5%; Enoxaparin Sodium; Nystatin Cream; Linezolid; Atorvastatin; Escitalopram Oxalate; Pantoprazole; traZODONE HCl; Tolterodine; Sodium Chloride 0.9%  Flush; Potassium Chloride; Furosemide; Vancomycin HCl; Fentanyl Citrate; Insulin Human Regular; NS; Zolpidem Tartrate; Acetylcysteine 20%; Fluticasone Propionate 110mcg; Hydralazine HCl; Lansoprazole Oral Suspension; Metoprolol; Acetaminophen (Liquid); Olanzapine (Disintegrating Tablet); Lorazepam; Levofloxacin; Ipratropium Bromide MDI; Albuterol; LR; Potassium Chloride Replacement (Oncology); Iso-Osmotic Dextrose; Neutra-Phos; Magnesium Sulfate; Calcium Gluconate; Propofol; Soln.; Metoclopramide; SW; Ketorolac
- Procedures performed: Other repair and plastic operations on trachea; Temporary tracheostomy; Continuous invasive mechanical ventilation for 96 consecutive hours or more; Other intubation of respiratory tract; Injection or infusion of oxazolidinone class of antibiotics; Insertion of endotracheal tube; Arterial catheterization; Venous catheterization, not elsewhere classified; Enteral infusion of concentrated nutritional substances; Parenteral infusion of concentrated nutritional substances
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 3839,
  "diagnoses_codes_current": "5191;5185;496;2762;45340;5533",
  "procedures_codes_current": "3179;311;9672;9605;14;9604;3891;3893;966;9915",
  "medications_current": "Vial;D5W;Pantoprazole Sodium;Levothyroxine Sodium;Heparin Flush PICC (100 units/ml);Acetaminophen;Heparin Sodium;Sodium Chloride Nasal;Sarna Lotion;Insulin;Multivitamin IV;Amino Acids 4.25% W/ Dextrose 5%;Enoxaparin Sodium;Nystatin Cream;Linezolid;Atorvastatin;Escitalopram Oxalate;Pantoprazole;traZODONE HCl;Tolterodine;Sodium Chloride 0.9%  Flush;Potassium Chloride;Furosemide;Vancomycin HCl;Fentanyl Citrate;Insulin Human Regular;NS;Zolpidem Tartrate;Acetylcysteine 20%;Fluticasone Propionate 110mcg;Hydralazine HCl;Lansoprazole Oral Suspension;Metoprolol;Acetaminophen (Liquid);Olanzapine (Disintegrating Tablet);Lorazepam;Levofloxacin;Ipratropium Bromide MDI;Albuterol;LR;Potassium Chloride Replacement (Oncology);Iso-Osmotic Dextrose;Neutra-Phos;Magnesium Sulfate;Calcium Gluconate;Propofol;Soln.;Metoclopramide;SW;Ketorolac;Guaifenesin;NS Epidural Bag;Hydromorphone P.F.;Bupivacaine 0.5%;Oxycodone-Acetaminophen;Hydromorphone;D5 1/2NS;Dolasetron Mesylate;Bisacodyl;Docusate Sodium;Dextrose 5%;Isosorbide Mononitrate (Extended Release);Heparin;Fluticasone-Salmeterol (250/50);Albuterol 0.083% Neb Soln;Ipratropium Bromide Neb",
  "label_lipid_next": 1,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 1,
  "label_kidney_next": 1,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: [ICD9-DX:5191]; [ICD9-DX:5185]; Chronic airway obstruction, not elsewhere classified; Acidosis; Acute venous embolism and thrombosis of unspecified deep vessels of lower extremity; Diaphragmatic hernia without mention of obstruction or gangrene\n- Medications prescribed: Vial; D5W; Pantoprazole Sodium; Levothyroxine Sodium; Heparin Flush PICC (100 units/ml); Acetaminophen; Heparin Sodium; Sodium Chloride Nasal; Sarna Lotion; Insulin; Multivitamin IV; Amino Acids 4.25% W/ Dextrose 5%; Enoxaparin Sodium; Nystatin Cream; Linezolid; Atorvastatin; Escitalopram Oxalate; Pantoprazole; traZODONE HCl; Tolterodine; Sodium Chloride 0.9%  Flush; Potassium Chloride; Furosemide; Vancomycin HCl; Fentanyl Citrate; Insulin Human Regular; NS; Zolpidem Tartrate; Acetylcysteine 20%; Fluticasone Propionate 110mcg; Hydralazine HCl; Lansoprazole Oral Suspension; Metoprolol; Acetaminophen (Liquid); Olanzapine (Disintegrating Tablet); Lorazepam; Levofloxacin; Ipratropium Bromide MDI; Albuterol; LR; Potassium Chloride Replacement (Oncology); Iso-Osmotic Dextrose; Neutra-Phos; Magnesium Sulfate; Calcium Gluconate; Propofol; Soln.; Metoclopramide; SW; Ketorolac\n- Procedures performed: Other repair and plastic operations on trachea; Temporary tracheostomy; Continuous invasive mechanical ventilation for 96 consecutive hours or more; Other intubation of respiratory tract; Injection or infusion of oxazolidinone class of antibiotics; Insertion of endotracheal tube; Arterial catheterization; Venous catheterization, not elsewhere classified; Enteral infusion of concentrated nutritional substances; Parenteral infusion of concentrated nutritional substances"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_014 — pair_id 4364

- **Tags:** cross_model_disagreement, cross_model_disagreement_high, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (4):** lipid, diabetes, hypertension, cardio
- **Ground truth (task → label):** `{'lipid': 1, 'diabetes': 0, 'hypertension': 0, 'obesity': 0, 'cardio': 0, 'kidney': 1, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['diabetes', 'hypertension', 'cardio', 'stroke']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.8}, 'diabetes': {'pred': 1, 'prob': 0.9}, 'hypertension': {'pred': 1, 'prob': 0.75}, 'obesity': {'pred': 0, 'prob': 0.1}, 'cardio': {'pred': 1, 'prob': 0.85}, 'kidney': {'pred': 1, 'prob': 0.9}, 'stroke': {'pred': 1, 'prob': 0.7}}`

**Reasoning:**

```
The patient is prescribed Atorvastatin, which is commonly used for lipid disorders, indicating a likely diagnosis of lipid metabolism issues. Insulin suggests current diabetes management. The presence of acute kidney failure indicates kidney issues. The patient has had an acute myocardial infarction, which raises the likelihood of cardiovascular disease and potential stroke risk. There is no evidence of obesity in the current visit record.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['lipid']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 41519;5849;4538;41081;00845;2765;42731;30390;55320. Current-visit narrative (truncated): - Diagnoses made: Other pulmonary embolism and infarction; Acute kidney failure, unspecified; [ICD9-DX:4538]; Acute myocardial infarction of other specified sites, initial episode of care; Intestinal infection due to Clostridium difficile; [ICD9-DX:2765]; Atrial fibrillation; Other and unspecified alcohol dependence, unspecified; Ventral, unspecified, hernia without mention of obstruction or gangrene - Medications prescribed: Insulin; Guaifenesin-Codeine Phosphate; Docusate Sodium; Acetaminophen; Aspirin EC; Folic Acid; Senna; Thiamine HCl; Sodium Chloride 0.9% Flush; Potassium Chloride; Diazepam; Magnesium Sulfate; Metoprolol; Warfarin; Lorazepam; NS; D5W; Heparin Sodium; Benzonatate; Guaifenesin; Metronidazole; Neutra-Phos; D5 1/2NS; Magnesium Oxide; Sodium Bicarbonate; Multivitamin IV; Heparin; Ceftriaxone; Atorvastatin; Iso-Osmotic Dextrose; Ipratropium Bromide Neb; Azithromycin; ...

### Current-visit narrative

```
- Diagnoses made: Other pulmonary embolism and infarction; Acute kidney failure, unspecified; [ICD9-DX:4538]; Acute myocardial infarction of other specified sites, initial episode of care; Intestinal infection due to Clostridium difficile; [ICD9-DX:2765]; Atrial fibrillation; Other and unspecified alcohol dependence, unspecified; Ventral, unspecified, hernia without mention of obstruction or gangrene
- Medications prescribed: Insulin; Guaifenesin-Codeine Phosphate; Docusate Sodium; Acetaminophen; Aspirin EC; Folic Acid; Senna; Thiamine HCl; Sodium Chloride 0.9%  Flush; Potassium Chloride; Diazepam; Magnesium Sulfate; Metoprolol; Warfarin; Lorazepam; NS; D5W; Heparin Sodium; Benzonatate; Guaifenesin; Metronidazole; Neutra-Phos; D5 1/2NS; Magnesium Oxide; Sodium Bicarbonate; Multivitamin IV; Heparin; Ceftriaxone; Atorvastatin; Iso-Osmotic Dextrose; Ipratropium Bromide Neb; Azithromycin; Acetylcysteine 20%
- Procedures performed: None recorded
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 4364,
  "diagnoses_codes_current": "41519;5849;4538;41081;00845;2765;42731;30390;55320",
  "procedures_codes_current": NaN,
  "medications_current": "Insulin;Guaifenesin-Codeine Phosphate;Docusate Sodium;Acetaminophen;Aspirin EC;Folic Acid;Senna;Thiamine HCl;Sodium Chloride 0.9%  Flush;Potassium Chloride;Diazepam;Magnesium Sulfate;Metoprolol;Warfarin;Lorazepam;NS;D5W;Heparin Sodium;Benzonatate;Guaifenesin;Metronidazole;Neutra-Phos;D5 1/2NS;Magnesium Oxide;Sodium Bicarbonate;Multivitamin IV;Heparin;Ceftriaxone;Atorvastatin;Iso-Osmotic Dextrose;Ipratropium Bromide Neb;Azithromycin;Acetylcysteine 20%",
  "label_lipid_next": 1,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 0,
  "label_kidney_next": 1,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Other pulmonary embolism and infarction; Acute kidney failure, unspecified; [ICD9-DX:4538]; Acute myocardial infarction of other specified sites, initial episode of care; Intestinal infection due to Clostridium difficile; [ICD9-DX:2765]; Atrial fibrillation; Other and unspecified alcohol dependence, unspecified; Ventral, unspecified, hernia without mention of obstruction or gangrene\n- Medications prescribed: Insulin; Guaifenesin-Codeine Phosphate; Docusate Sodium; Acetaminophen; Aspirin EC; Folic Acid; Senna; Thiamine HCl; Sodium Chloride 0.9%  Flush; Potassium Chloride; Diazepam; Magnesium Sulfate; Metoprolol; Warfarin; Lorazepam; NS; D5W; Heparin Sodium; Benzonatate; Guaifenesin; Metronidazole; Neutra-Phos; D5 1/2NS; Magnesium Oxide; Sodium Bicarbonate; Multivitamin IV; Heparin; Ceftriaxone; Atorvastatin; Iso-Osmotic Dextrose; Ipratropium Bromide Neb; Azithromycin; Acetylcysteine 20%\n- Procedures performed: None recorded"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_015 — pair_id 5346

- **Tags:** cross_model_disagreement, cross_model_disagreement_high, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (4):** lipid, diabetes, hypertension, cardio
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 0, 'obesity': 0, 'cardio': 1, 'kidney': 1, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['lipid', 'diabetes', 'hypertension', 'stroke']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.8}, 'diabetes': {'pred': 1, 'prob': 0.9}, 'hypertension': {'pred': 1, 'prob': 0.85}, 'obesity': {'pred': 0, 'prob': 0.1}, 'cardio': {'pred': 1, 'prob': 0.75}, 'kidney': {'pred': 1, 'prob': 0.7}, 'stroke': {'pred': 1, 'prob': 0.65}}`

**Reasoning:**

```
The patient is prescribed Atorvastatin, which indicates a history of lipid disorder. Insulin use suggests current diabetes. The presence of congestive heart failure and atrial fibrillation indicates hypertension. There is no evidence of obesity in the record. Given the patient's cardiovascular issues, there is a high likelihood of future cardiovascular disease, kidney disease due to heart failure, and stroke risk due to atrial fibrillation.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 99662;0389;2851;2875;56985;42731;4280;V4581;V4501. Current-visit narrative (truncated): - Diagnoses made: Infection and inflammatory reaction due to other vascular device, implant, and graft; Unspecified septicemia; Acute posthemorrhagic anemia; Thrombocytopenia, unspecified; Angiodysplasia of intestine with hemorrhage; Atrial fibrillation; Congestive heart failure, unspecified; Aortocoronary bypass status; Cardiac pacemaker in situ - Medications prescribed: Vancomycin HCl; Insulin; Epoetin Alfa; Vial; D5W; Potassium Chloride; Bumetanide; Furosemide; Bivalirudin; Lorazepam; Gentamicin; Isotonic Sodium Chloride; Magnesium Sulfate; Magnesium Oxide; Iso-Osmotic Dextrose; Aspirin EC; Meropenem; NS; SW; Dopamine HCl; Adenosine; Ibutilide Fumarate; Sarna Lotion; traZODONE HCl; Warfarin; Imipenem-Cilastatin; Sertraline HCl; Carvedilol; Enalapril Maleate; Ciprofloxacin; Metronidazole; Miconazole Powder 2%; Docusate Sodium; Senna; Atorvastatin; Amiodarone HCl; Acetaminophen; Pant...

### Current-visit narrative

```
- Diagnoses made: Infection and inflammatory reaction due to other vascular device, implant, and graft; Unspecified septicemia; Acute posthemorrhagic anemia; Thrombocytopenia, unspecified; Angiodysplasia of intestine with hemorrhage; Atrial fibrillation; Congestive heart failure, unspecified; Aortocoronary bypass status; Cardiac pacemaker in situ
- Medications prescribed: Vancomycin HCl; Insulin; Epoetin Alfa; Vial; D5W; Potassium Chloride; Bumetanide; Furosemide; Bivalirudin; Lorazepam; Gentamicin; Isotonic Sodium Chloride; Magnesium Sulfate; Magnesium Oxide; Iso-Osmotic Dextrose; Aspirin EC; Meropenem; NS; SW; Dopamine HCl; Adenosine; Ibutilide Fumarate; Sarna Lotion; traZODONE HCl; Warfarin; Imipenem-Cilastatin; Sertraline HCl; Carvedilol; Enalapril Maleate; Ciprofloxacin; Metronidazole; Miconazole Powder 2%; Docusate Sodium; Senna; Atorvastatin; Amiodarone HCl; Acetaminophen; Pantoprazole; Aluminum-Magnesium Hydroxide; Golytely; Potassium Phosphate; Ranitidine; DopAmine; Diphenhydramine HCl; Linezolid
- Procedures performed: Other endoscopy of small intestine; Endoscopic control of gastric or duodenal bleeding; Colonoscopy; Other electric countershock of heart; Diagnostic ultrasound of heart; Venous catheterization, not elsewhere classified
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 5346,
  "diagnoses_codes_current": "99662;0389;2851;2875;56985;42731;4280;V4581;V4501",
  "procedures_codes_current": "4513;4443;4523;9962;8872;3893",
  "medications_current": "Vancomycin HCl;Insulin;Epoetin Alfa;Vial;D5W;Potassium Chloride;Bumetanide;Furosemide;Bivalirudin;Lorazepam;Gentamicin;Isotonic Sodium Chloride;Magnesium Sulfate;Magnesium Oxide;Iso-Osmotic Dextrose;Aspirin EC;Meropenem;NS;SW;Dopamine HCl;Adenosine;Ibutilide Fumarate;Sarna Lotion;traZODONE HCl;Warfarin;Imipenem-Cilastatin;Sertraline HCl;Carvedilol;Enalapril Maleate;Ciprofloxacin;Metronidazole;Miconazole Powder 2%;Docusate Sodium;Senna;Atorvastatin;Amiodarone HCl;Acetaminophen;Pantoprazole;Aluminum-Magnesium Hydroxide;Golytely;Potassium Phosphate;Ranitidine;DopAmine;Diphenhydramine HCl;Linezolid",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 1,
  "label_kidney_next": 1,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Infection and inflammatory reaction due to other vascular device, implant, and graft; Unspecified septicemia; Acute posthemorrhagic anemia; Thrombocytopenia, unspecified; Angiodysplasia of intestine with hemorrhage; Atrial fibrillation; Congestive heart failure, unspecified; Aortocoronary bypass status; Cardiac pacemaker in situ\n- Medications prescribed: Vancomycin HCl; Insulin; Epoetin Alfa; Vial; D5W; Potassium Chloride; Bumetanide; Furosemide; Bivalirudin; Lorazepam; Gentamicin; Isotonic Sodium Chloride; Magnesium Sulfate; Magnesium Oxide; Iso-Osmotic Dextrose; Aspirin EC; Meropenem; NS; SW; Dopamine HCl; Adenosine; Ibutilide Fumarate; Sarna Lotion; traZODONE HCl; Warfarin; Imipenem-Cilastatin; Sertraline HCl; Carvedilol; Enalapril Maleate; Ciprofloxacin; Metronidazole; Miconazole Powder 2%; Docusate Sodium; Senna; Atorvastatin; Amiodarone HCl; Acetaminophen; Pantoprazole; Aluminum-Magnesium Hydroxide; Golytely; Potassium Phosphate; Ranitidine; DopAmine; Diphenhydramine HCl; Linezolid\n- Procedures performed: Other endoscopy of small intestine; Endoscopic control of gastric or duodenal bleeding; Colonoscopy; Other electric countershock of heart; Diagnostic ultrasound of heart; Venous catheterization, not elsewhere classified"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_016 — pair_id 5493

- **Tags:** cross_model_disagreement, cross_model_disagreement_high, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (4):** lipid, diabetes, hypertension, cardio
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 0, 'obesity': 0, 'cardio': 1, 'kidney': 1, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['lipid', 'diabetes', 'hypertension', 'stroke']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.8}, 'diabetes': {'pred': 1, 'prob': 0.9}, 'hypertension': {'pred': 1, 'prob': 0.85}, 'obesity': {'pred': 0, 'prob': 0.1}, 'cardio': {'pred': 1, 'prob': 0.75}, 'kidney': {'pred': 1, 'prob': 0.7}, 'stroke': {'pred': 1, 'prob': 0.65}}`

**Reasoning:**

```
The patient is on Atorvastatin, which suggests a history of lipid disorders. Insulin indicates current diabetes management. Lisinopril and Metoprolol suggest hypertension treatment. There is no evidence of obesity in the record. The patient's history of multiple serious conditions, including aortocoronary bypass, suggests a risk for cardiovascular disease. The presence of urinary tract infection and other complications may indicate a risk for kidney disease. The patient's neurological issues and history of embolism raise concerns for potential stroke.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 3239;340;41519;99664;5990;4538;3441;56400;78907;34120;6019;70901;V4581;49390;V5865. Current-visit narrative (truncated): - Diagnoses made: Unspecified causes of encephalitis, myelitis, and encephalomyelitis; Multiple sclerosis; Other pulmonary embolism and infarction; Infection and inflammatory reaction due to indwelling urinary catheter; Urinary tract infection, site not specified; [ICD9-DX:4538]; Paraplegia; Constipation, unspecified; Abdominal pain, generalized; Acute (transverse) myelitis NOS; Prostatitis, unspecified; Vitiligo; Aortocoronary bypass status; Asthma, unspecified type, unspecified; Long-term (current) use of steroids - Medications prescribed: Heparin Sodium; D5W; MethylPREDNISolone Sodium Succ; traZODONE HCl; Golytely; Insulin; NS; Lorazepam; Warfarin; Mannitol 20%; Soln.; Potassium Chloride; Pantoprazole; Glycerin Supps; Debrox 6.5%; Prednisone; Acetaminophen; traMADOL; Lactulose; Fluticasone-Salmeterol (250/50); Lisinopril; Bisacodyl; Dexamethasone; Heparin; Milk of Magnesia; Aspirin...

### Current-visit narrative

```
- Diagnoses made: Unspecified causes of encephalitis, myelitis, and encephalomyelitis; Multiple sclerosis; Other pulmonary embolism and infarction; Infection and inflammatory reaction due to indwelling urinary catheter; Urinary tract infection, site not specified; [ICD9-DX:4538]; Paraplegia; Constipation, unspecified; Abdominal pain, generalized; Acute (transverse) myelitis NOS; Prostatitis, unspecified; Vitiligo; Aortocoronary bypass status; Asthma, unspecified type, unspecified; Long-term (current) use of steroids
- Medications prescribed: Heparin Sodium; D5W; MethylPREDNISolone Sodium Succ; traZODONE HCl; Golytely; Insulin; NS; Lorazepam; Warfarin; Mannitol 20%; Soln.; Potassium Chloride; Pantoprazole; Glycerin Supps; Debrox 6.5%; Prednisone; Acetaminophen; traMADOL; Lactulose; Fluticasone-Salmeterol (250/50); Lisinopril; Bisacodyl; Dexamethasone; Heparin; Milk of Magnesia; Aspirin; Sodium Chloride 0.9%  Flush; Docusate Sodium; Metoprolol XL; Atorvastatin; Montelukast Sodium; Tuberculin Protein; Senna; CefTRIAXone; Vancomycin HCl; Iso-Osmotic Dextrose; Fludrocortisone Acetate; Potassium Chl 40 mEq / 1000 mL NS; Miconazole Powder 2%; Pneumococcal Vac Polyvalent; Gabapentin; Magnesium Sulfate; Oxycodone-Acetaminophen; Phenytoin Sodium; Potassium Chl 20 mEq / 1000 mL NS; NS (Glass Bottle); Phenytoin; Azithromycin; Levofloxacin; CeftAZIDime
- Procedures performed: Open biopsy of brain; Interruption of the vena cava; Incision of lung; Venous catheterization, not elsewhere classified
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 5493,
  "diagnoses_codes_current": "3239;340;41519;99664;5990;4538;3441;56400;78907;34120;6019;70901;V4581;49390;V5865",
  "procedures_codes_current": "114;387;331;3893",
  "medications_current": "Heparin Sodium;D5W;MethylPREDNISolone Sodium Succ;traZODONE HCl;Golytely;Insulin;NS;Lorazepam;Warfarin;Mannitol 20%;Soln.;Potassium Chloride;Pantoprazole;Glycerin Supps;Debrox 6.5%;Prednisone;Acetaminophen;traMADOL;Lactulose;Fluticasone-Salmeterol (250/50);Lisinopril;Bisacodyl;Dexamethasone;Heparin;Milk of Magnesia;Aspirin;Sodium Chloride 0.9%  Flush;Docusate Sodium;Metoprolol XL;Atorvastatin;Montelukast Sodium;Tuberculin Protein;Senna;CefTRIAXone;Vancomycin HCl;Iso-Osmotic Dextrose;Fludrocortisone Acetate;Potassium Chl 40 mEq / 1000 mL NS;Miconazole Powder 2%;Pneumococcal Vac Polyvalent;Gabapentin;Magnesium Sulfate;Oxycodone-Acetaminophen;Phenytoin Sodium;Potassium Chl 20 mEq / 1000 mL NS;NS (Glass Bottle);Phenytoin;Azithromycin;Levofloxacin;CeftAZIDime;Hydrocortisone Na Succ.;Metoprolol;Lidocaine 5% Patch;Magnesium Citrate;Simethicone;Advair Diskus;Alprazolam;Tizanidine HCl;Ciprofloxacin HCl;Albuterol",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 1,
  "label_kidney_next": 1,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Unspecified causes of encephalitis, myelitis, and encephalomyelitis; Multiple sclerosis; Other pulmonary embolism and infarction; Infection and inflammatory reaction due to indwelling urinary catheter; Urinary tract infection, site not specified; [ICD9-DX:4538]; Paraplegia; Constipation, unspecified; Abdominal pain, generalized; Acute (transverse) myelitis NOS; Prostatitis, unspecified; Vitiligo; Aortocoronary bypass status; Asthma, unspecified type, unspecified; Long-term (current) use of steroids\n- Medications prescribed: Heparin Sodium; D5W; MethylPREDNISolone Sodium Succ; traZODONE HCl; Golytely; Insulin; NS; Lorazepam; Warfarin; Mannitol 20%; Soln.; Potassium Chloride; Pantoprazole; Glycerin Supps; Debrox 6.5%; Prednisone; Acetaminophen; traMADOL; Lactulose; Fluticasone-Salmeterol (250/50); Lisinopril; Bisacodyl; Dexamethasone; Heparin; Milk of Magnesia; Aspirin; Sodium Chloride 0.9%  Flush; Docusate Sodium; Metoprolol XL; Atorvastatin; Montelukast Sodium; Tuberculin Protein; Senna; CefTRIAXone; Vancomycin HCl; Iso-Osmotic Dextrose; Fludrocortisone Acetate; Potassium Chl 40 mEq / 1000 mL NS; Miconazole Powder 2%; Pneumococcal Vac Polyvalent; Gabapentin; Magnesium Sulfate; Oxycodone-Acetaminophen; Phenytoin Sodium; Potassium Chl 20 mEq / 1000 mL NS; NS (Glass Bottle); Phenytoin; Azithromycin; Levofloxacin; CeftAZIDime\n- Procedures performed: Open biopsy of brain; Interruption of the vena cava; Incision of lung; Venous catheterization, not elsewhere classified"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_017 — pair_id 5999

- **Tags:** cross_model_disagreement, cross_model_disagreement_high, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (4):** lipid, diabetes, hypertension, cardio
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 0, 'obesity': 0, 'cardio': 1, 'kidney': 0, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['lipid', 'diabetes', 'hypertension', 'kidney', 'stroke']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.8}, 'diabetes': {'pred': 1, 'prob': 0.9}, 'hypertension': {'pred': 1, 'prob': 0.85}, 'obesity': {'pred': 0, 'prob': 0.1}, 'cardio': {'pred': 1, 'prob': 0.95}, 'kidney': {'pred': 1, 'prob': 0.9}, 'stroke': {'pred': 1, 'prob': 0.75}}`

**Reasoning:**

```
The patient has multiple cardiovascular issues, including coronary atherosclerosis and heart failure, which suggest a high likelihood of lipid metabolism disorders. Insulin use indicates diabetes. The presence of cardiogenic shock and heart failure suggests hypertension. There is no evidence of obesity in the record. Given the acute kidney failure, kidney disease is likely to persist. The history of cerebrovascular infarction indicates a risk for stroke.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 41061;78551;4280;4240;99702;5845;41401;99609;5119. Current-visit narrative (truncated): - Diagnoses made: True posterior wall infarction, initial episode of care; Cardiogenic shock; Congestive heart failure, unspecified; Mitral valve disorders; Iatrogenic cerebrovascular infarction or hemorrhage; Acute kidney failure with lesion of tubular necrosis; Coronary atherosclerosis of native coronary artery; Other mechanical complication of cardiac device, implant, and graft; Unspecified pleural effusion - Medications prescribed: Diphenhydramine HCl; D5W; Metolazone; Nesiritide; Insulin; Metoprolol; Lorazepam; Haloperidol; Phytonadione; Nicotine; Nystatin Ointment; Hydralazine HCl; Aspirin; Docusate Sodium (Liquid); Fentanyl Citrate; Levofloxacin; Midazolam HCl; Dextrose 5%; Furosemide; Quetiapine Fumarate; Propofol (Generic); DopAmine; Fluconazole; Acetazolamide Sodium; SW; Potassium Chloride; Magnesium Sulfate; Nitroprusside Sodium; Phenylephrine HCl; NS; Metoclopramide; Nitro...

### Current-visit narrative

```
- Diagnoses made: True posterior wall infarction, initial episode of care; Cardiogenic shock; Congestive heart failure, unspecified; Mitral valve disorders; Iatrogenic cerebrovascular infarction or hemorrhage; Acute kidney failure with lesion of tubular necrosis; Coronary atherosclerosis of native coronary artery; Other mechanical complication of cardiac device, implant, and graft; Unspecified pleural effusion
- Medications prescribed: Diphenhydramine HCl; D5W; Metolazone; Nesiritide; Insulin; Metoprolol; Lorazepam; Haloperidol; Phytonadione; Nicotine; Nystatin Ointment; Hydralazine HCl; Aspirin; Docusate Sodium (Liquid); Fentanyl Citrate; Levofloxacin; Midazolam HCl; Dextrose 5%; Furosemide; Quetiapine Fumarate; Propofol (Generic); DopAmine; Fluconazole; Acetazolamide Sodium; SW; Potassium Chloride; Magnesium Sulfate; Nitroprusside Sodium; Phenylephrine HCl; NS; Metoclopramide; Nitroglycerin; Humulin-R Insulin; Morphine Sulfate; Aspirin EC; Docusate Sodium; Calcium Gluconate; Oxycodone-Acetaminophen; Acetaminophen; Hespan; Lansoprazole Oral Solution; Albuterol; Iso-Osmotic Dextrose; Vancomycin HCl; Heparin Sodium; Atropine Sulfate; Pantoprazole; Neutra-Phos; Atorvastatin; D5W (GLASS BOTTLE)
- Procedures performed: Single internal mammary-coronary artery bypass; Combined right and left heart cardiac catheterization; Coronary arteriography using two catheters; Implant of pulsation balloon; Diagnostic ultrasound of heart; Injection or infusion of nesiritide; (Aorto)coronary bypass of two coronary arteries; Extracorporeal circulation auxiliary to open heart surgery; Thoracentesis; Suture of artery; Insertion of endotracheal tube; Fiber-optic bronchoscopy; Enteral infusion of concentrated nutritional substances
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 5999,
  "diagnoses_codes_current": "41061;78551;4280;4240;99702;5845;41401;99609;5119",
  "procedures_codes_current": "3615;3723;8856;3761;8872;13;3612;3961;3491;3931;9604;3322;966",
  "medications_current": "Diphenhydramine HCl;D5W;Metolazone;Nesiritide;Insulin;Metoprolol;Lorazepam;Haloperidol;Phytonadione;Nicotine;Nystatin Ointment;Hydralazine HCl;Aspirin;Docusate Sodium (Liquid);Fentanyl Citrate;Levofloxacin;Midazolam HCl;Dextrose 5%;Furosemide;Quetiapine Fumarate;Propofol (Generic);DopAmine;Fluconazole;Acetazolamide Sodium;SW;Potassium Chloride;Magnesium Sulfate;Nitroprusside Sodium;Phenylephrine HCl;NS;Metoclopramide;Nitroglycerin;Humulin-R Insulin;Morphine Sulfate;Aspirin EC;Docusate Sodium;Calcium Gluconate;Oxycodone-Acetaminophen;Acetaminophen;Hespan;Lansoprazole Oral Solution;Albuterol;Iso-Osmotic Dextrose;Vancomycin HCl;Heparin Sodium;Atropine Sulfate;Pantoprazole;Neutra-Phos;Atorvastatin;D5W (GLASS BOTTLE);Protamine Sulfate;Ranitidine;Soln.;Meperidine;Sucralfate;Neostigmine;Glycopyrrolate;Dobutamine",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 1,
  "label_kidney_next": 0,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: True posterior wall infarction, initial episode of care; Cardiogenic shock; Congestive heart failure, unspecified; Mitral valve disorders; Iatrogenic cerebrovascular infarction or hemorrhage; Acute kidney failure with lesion of tubular necrosis; Coronary atherosclerosis of native coronary artery; Other mechanical complication of cardiac device, implant, and graft; Unspecified pleural effusion\n- Medications prescribed: Diphenhydramine HCl; D5W; Metolazone; Nesiritide; Insulin; Metoprolol; Lorazepam; Haloperidol; Phytonadione; Nicotine; Nystatin Ointment; Hydralazine HCl; Aspirin; Docusate Sodium (Liquid); Fentanyl Citrate; Levofloxacin; Midazolam HCl; Dextrose 5%; Furosemide; Quetiapine Fumarate; Propofol (Generic); DopAmine; Fluconazole; Acetazolamide Sodium; SW; Potassium Chloride; Magnesium Sulfate; Nitroprusside Sodium; Phenylephrine HCl; NS; Metoclopramide; Nitroglycerin; Humulin-R Insulin; Morphine Sulfate; Aspirin EC; Docusate Sodium; Calcium Gluconate; Oxycodone-Acetaminophen; Acetaminophen; Hespan; Lansoprazole Oral Solution; Albuterol; Iso-Osmotic Dextrose; Vancomycin HCl; Heparin Sodium; Atropine Sulfate; Pantoprazole; Neutra-Phos; Atorvastatin; D5W (GLASS BOTTLE)\n- Procedures performed: Single internal mammary-coronary artery bypass; Combined right and left heart cardiac catheterization; Coronary arteriography using two catheters; Implant of pulsation balloon; Diagnostic ultrasound of heart; Injection or infusion of nesiritide; (Aorto)coronary bypass of two coronary arteries; Extracorporeal circulation auxiliary to open heart surgery; Thoracentesis; Suture of artery; Insertion of endotracheal tube; Fiber-optic bronchoscopy; Enteral infusion of concentrated nutritional substances"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_018 — pair_id 6864

- **Tags:** cross_model_disagreement, cross_model_disagreement_high, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (3):** lipid, hypertension, cardio
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 1, 'hypertension': 0, 'obesity': 0, 'cardio': 1, 'kidney': 0, 'stroke': 1}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['hypertension', 'stroke']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 1, 'prob': 1.0}, 'hypertension': {'pred': 1, 'prob': 1.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.5}, 'kidney': {'pred': 0, 'prob': 0.0}, 'stroke': {'pred': 0, 'prob': 0.0}}`

**Reasoning:**

```
The patient has a documented diagnosis of diabetes mellitus and hypertension. There is no evidence of lipid disorders, obesity, kidney disease, or stroke in the current visit record. Cardiovascular disease is a possibility due to the presence of hypertension and other conditions, but there is no explicit diagnosis.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['lipid', 'cardio']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.95}, 'diabetes': {'pred': 1, 'prob': 0.95}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 56989;5772;45341;2765;56962;5723;5771;25000;V632;V1041. Current-visit narrative (truncated): - Diagnoses made: Other specified disorders of intestine; Cyst and pseudocyst of pancreas; Acute venous embolism and thrombosis of deep vessels of proximal lower extremity; [ICD9-DX:2765]; Mechanical complication of colostomy and enterostomy; Portal hypertension; Chronic pancreatitis; Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; Person awaiting admission to adequate facility elsewhere; Personal history of malignant neoplasm of cervix uteri - Medications prescribed: Alteplase (Catheter Clearance); Magnesium Sulfate; Potassium Chloride; NS; Insulin; Sodium Chloride 0.9% Flush; SW; D5W; Calcium Gluconate; Potassium Phosphate; Heparin; Levofloxacin; Dextrose 5%; Metoprolol; Dolasetron Mesylate; Lidocaine 1%; Insulin Human Regular; Octreotide Acetate; AMP; Hydromorphone; Gabapentin; Ketorolac; Multivitamin IV; Amino Acids 4.25%...

### Current-visit narrative

```
- Diagnoses made: Other specified disorders of intestine; Cyst and pseudocyst of pancreas; Acute venous embolism and thrombosis of deep vessels of proximal lower extremity; [ICD9-DX:2765]; Mechanical complication of colostomy and enterostomy; Portal hypertension; Chronic pancreatitis; Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; Person awaiting admission to adequate facility elsewhere; Personal history of malignant neoplasm of cervix uteri
- Medications prescribed: Alteplase (Catheter Clearance); Magnesium Sulfate; Potassium Chloride; NS; Insulin; Sodium Chloride 0.9%  Flush; SW; D5W; Calcium Gluconate; Potassium Phosphate; Heparin; Levofloxacin; Dextrose 5%; Metoprolol; Dolasetron Mesylate; Lidocaine 1%; Insulin Human Regular; Octreotide Acetate; AMP; Hydromorphone; Gabapentin; Ketorolac; Multivitamin IV; Amino Acids 4.25% W/ Dextrose 5%; Metronidazole; Vancomycin HCl; Iso-Osmotic Dextrose; Cholestyramine; Atenolol; LR; Heparin Sodium; Heparin Flush CVL  (100 units/ml); Acetylcysteine 20%; Linezolid; Pantoprazole Sodium; Artificial Tears; Vial; Potassium Chl 20 mEq / 1000 mL D5NS; Miconazole Powder 2%; D5 1/2NS; Famotidine; Nystatin Oral Suspension; BuPROPion; Calcium Carbonate; Ciprofloxacin HCl; Diltiazem
- Procedures performed: Interruption of the vena cava; Parenteral infusion of concentrated nutritional substances; Injection or infusion of oxazolidinone class of antibiotics; Pulmonary artery wedge monitoring; Venous catheterization, not elsewhere classified; Arterial catheterization; Replacement of tube or enterostomy device of small intestine
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 6864,
  "diagnoses_codes_current": "56989;5772;45341;2765;56962;5723;5771;25000;V632;V1041",
  "procedures_codes_current": "387;9915;14;8964;3893;3891;9703",
  "medications_current": "Alteplase (Catheter Clearance);Magnesium Sulfate;Potassium Chloride;NS;Insulin;Sodium Chloride 0.9%  Flush;SW;D5W;Calcium Gluconate;Potassium Phosphate;Heparin;Levofloxacin;Dextrose 5%;Metoprolol;Dolasetron Mesylate;Lidocaine 1%;Insulin Human Regular;Octreotide Acetate;AMP;Hydromorphone;Gabapentin;Ketorolac;Multivitamin IV;Amino Acids 4.25% W/ Dextrose 5%;Metronidazole;Vancomycin HCl;Iso-Osmotic Dextrose;Cholestyramine;Atenolol;LR;Heparin Sodium;Heparin Flush CVL  (100 units/ml);Acetylcysteine 20%;Linezolid;Pantoprazole Sodium;Artificial Tears;Vial;Potassium Chl 20 mEq / 1000 mL D5NS;Miconazole Powder 2%;D5 1/2NS;Famotidine;Nystatin Oral Suspension;BuPROPion;Calcium Carbonate;Ciprofloxacin HCl;Diltiazem",
  "label_lipid_next": 0,
  "label_diabetes_current": 1,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 1,
  "label_kidney_next": 0,
  "label_stroke_next": 1,
  "narrative_current": "- Diagnoses made: Other specified disorders of intestine; Cyst and pseudocyst of pancreas; Acute venous embolism and thrombosis of deep vessels of proximal lower extremity; [ICD9-DX:2765]; Mechanical complication of colostomy and enterostomy; Portal hypertension; Chronic pancreatitis; Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; Person awaiting admission to adequate facility elsewhere; Personal history of malignant neoplasm of cervix uteri\n- Medications prescribed: Alteplase (Catheter Clearance); Magnesium Sulfate; Potassium Chloride; NS; Insulin; Sodium Chloride 0.9%  Flush; SW; D5W; Calcium Gluconate; Potassium Phosphate; Heparin; Levofloxacin; Dextrose 5%; Metoprolol; Dolasetron Mesylate; Lidocaine 1%; Insulin Human Regular; Octreotide Acetate; AMP; Hydromorphone; Gabapentin; Ketorolac; Multivitamin IV; Amino Acids 4.25% W/ Dextrose 5%; Metronidazole; Vancomycin HCl; Iso-Osmotic Dextrose; Cholestyramine; Atenolol; LR; Heparin Sodium; Heparin Flush CVL  (100 units/ml); Acetylcysteine 20%; Linezolid; Pantoprazole Sodium; Artificial Tears; Vial; Potassium Chl 20 mEq / 1000 mL D5NS; Miconazole Powder 2%; D5 1/2NS; Famotidine; Nystatin Oral Suspension; BuPROPion; Calcium Carbonate; Ciprofloxacin HCl; Diltiazem\n- Procedures performed: Interruption of the vena cava; Parenteral infusion of concentrated nutritional substances; Injection or infusion of oxazolidinone class of antibiotics; Pulmonary artery wedge monitoring; Venous catheterization, not elsewhere classified; Arterial catheterization; Replacement of tube or enterostomy device of small intestine"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_019 — pair_id 7366

- **Tags:** cross_model_disagreement, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (2):** lipid, cardio
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 1, 'hypertension': 1, 'obesity': 0, 'cardio': 0, 'kidney': 1, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['lipid', 'cardio']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.8}, 'diabetes': {'pred': 1, 'prob': 1.0}, 'hypertension': {'pred': 1, 'prob': 1.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.7}, 'kidney': {'pred': 1, 'prob': 0.8}, 'stroke': {'pred': 0, 'prob': 0.1}}`

**Reasoning:**

```
The patient has a documented diagnosis of diabetes, hypertension, and an unspecified disorder of kidney and ureter. Atorvastatin is prescribed, which suggests a concern for lipid metabolism disorders. There is no evidence of obesity in the record. The risk for cardiovascular disease is indicated by the hypertension and diabetes. The kidney disorder suggests a likelihood of ongoing kidney issues. There is no evidence to support a prediction of stroke.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: []
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 1, 'prob': 0.0}, 'hypertension': {'pred': 1, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 25011;2765;2859;4019;5939. Current-visit narrative (truncated): - Diagnoses made: Diabetes with ketoacidosis, type I [juvenile type], not stated as uncontrolled; [ICD9-DX:2765]; Anemia, unspecified; Unspecified essential hypertension; Unspecified disorder of kidney and ureter - Medications prescribed: Atenolol; Humulin-R Insulin; NS; Insulin; Metoprolol; Atorvastatin; Losartan Potassium; Cyanocobalamin; Heparin; Bisacodyl; Docusate Sodium; Acetaminophen; Neutra-Phos; Potassium Chloride; Calcium Carbonate; Magnesium Sulfate; Aspirin EC; Nitroglycerin SL; Zolpidem Tartrate - Procedures performed: Arterial catheterization

### Current-visit narrative

```
- Diagnoses made: Diabetes with ketoacidosis, type I [juvenile type], not stated as uncontrolled; [ICD9-DX:2765]; Anemia, unspecified; Unspecified essential hypertension; Unspecified disorder of kidney and ureter
- Medications prescribed: Atenolol; Humulin-R Insulin; NS; Insulin; Metoprolol; Atorvastatin; Losartan Potassium; Cyanocobalamin; Heparin; Bisacodyl; Docusate Sodium; Acetaminophen; Neutra-Phos; Potassium Chloride; Calcium Carbonate; Magnesium Sulfate; Aspirin EC; Nitroglycerin SL; Zolpidem Tartrate
- Procedures performed: Arterial catheterization
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 7366,
  "diagnoses_codes_current": "25011;2765;2859;4019;5939",
  "procedures_codes_current": "3891",
  "medications_current": "Atenolol;Humulin-R Insulin;NS;Insulin;Metoprolol;Atorvastatin;Losartan Potassium;Cyanocobalamin;Heparin;Bisacodyl;Docusate Sodium;Acetaminophen;Neutra-Phos;Potassium Chloride;Calcium Carbonate;Magnesium Sulfate;Aspirin EC;Nitroglycerin SL;Zolpidem Tartrate",
  "label_lipid_next": 0,
  "label_diabetes_current": 1,
  "label_hypertension_current": 1,
  "label_obesity_current": 0,
  "label_cardio_next": 0,
  "label_kidney_next": 1,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Diabetes with ketoacidosis, type I [juvenile type], not stated as uncontrolled; [ICD9-DX:2765]; Anemia, unspecified; Unspecified essential hypertension; Unspecified disorder of kidney and ureter\n- Medications prescribed: Atenolol; Humulin-R Insulin; NS; Insulin; Metoprolol; Atorvastatin; Losartan Potassium; Cyanocobalamin; Heparin; Bisacodyl; Docusate Sodium; Acetaminophen; Neutra-Phos; Potassium Chloride; Calcium Carbonate; Magnesium Sulfate; Aspirin EC; Nitroglycerin SL; Zolpidem Tartrate\n- Procedures performed: Arterial catheterization"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_020 — pair_id 7461

- **Tags:** cross_model_disagreement, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (1):** lipid
- **Ground truth (task → label):** `{'lipid': 1, 'diabetes': 1, 'hypertension': 0, 'obesity': 1, 'cardio': 1, 'kidney': 0, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['lipid']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 1, 'prob': 1.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 1, 'prob': 1.0}, 'cardio': {'pred': 1, 'prob': 0.75}, 'kidney': {'pred': 0, 'prob': 0.0}, 'stroke': {'pred': 0, 'prob': 0.0}}`

**Reasoning:**

```
The patient has a documented diagnosis of diabetes mellitus and morbid obesity. There is no evidence of hypertension in the current visit record. The patient has a history of cardiovascular issues (aortocoronary bypass status and old myocardial infarction), which suggests a likelihood of cardiovascular disease in the next visit. There is no evidence of kidney disease or stroke in the current visit.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: []
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.95}, 'diabetes': {'pred': 1, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 1, 'prob': 0.95}, 'cardio': {'pred': 1, 'prob': 0.95}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 0389;70703;73008;5990;2639;99591;25000;3558;3441;78057;78791;7245;V4581;412;27801. Current-visit narrative (truncated): - Diagnoses made: Unspecified septicemia; Pressure ulcer, lower back; Acute osteomyelitis, other specified sites; Urinary tract infection, site not specified; Unspecified protein-calorie malnutrition; Sepsis; Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; Mononeuritis of lower limb, unspecified; Paraplegia; Unspecified sleep apnea; Diarrhea; Backache, unspecified; Aortocoronary bypass status; Old myocardial infarction; Morbid obesity - Medications prescribed: Daptomycin; Alteplase (Catheter Clearance); Tigecycline; OxycoDONE (Immediate Release); Acetaminophen; Insulin; Vancomycin; Docusate Sodium; Senna; Iso-Osmotic Dextrose; Nicotine Patch; Pantoprazole; Simvastatin; Ipratropium Bromide MDI; Losartan Potassium; Heparin; Furosemide; Fentanyl Patch; Aspirin EC; Albuterol; Sodium Chloride 0.9% Flush; 1/2 NS; NS (Mini Bag Plus)...

### Current-visit narrative

```
- Diagnoses made: Unspecified septicemia; Pressure ulcer, lower back; Acute osteomyelitis, other specified sites; Urinary tract infection, site not specified; Unspecified protein-calorie malnutrition; Sepsis; Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; Mononeuritis of lower limb, unspecified; Paraplegia; Unspecified sleep apnea; Diarrhea; Backache, unspecified; Aortocoronary bypass status; Old myocardial infarction; Morbid obesity
- Medications prescribed: Daptomycin; Alteplase (Catheter Clearance); Tigecycline; OxycoDONE (Immediate Release); Acetaminophen; Insulin; Vancomycin; Docusate Sodium; Senna; Iso-Osmotic Dextrose; Nicotine Patch; Pantoprazole; Simvastatin; Ipratropium Bromide MDI; Losartan Potassium; Heparin; Furosemide; Fentanyl Patch; Aspirin EC; Albuterol; Sodium Chloride 0.9%  Flush; 1/2 NS; NS (Mini Bag Plus); Meropenem; Metoprolol; Nystatin Oral Suspension; Oxycodone-Acetaminophen; D5 1/2NS; Calcium Gluconate; NS; Dakins 1/4 Strength; LR; Nafcillin; Piperacillin-Tazobactam Na; Docusate Sodium (Liquid); MetRONIDAZOLE (FLagyl); Potassium Chloride; Enoxaparin Sodium; D5W; Phenylephrine; Aspirin; Phytonadione; Heparin Flush CVL  (100 units/ml); Famotidine; Bisacodyl; DiphenhydrAMINE; Zinc Sulfate; Multivitamins W/minerals; Ascorbic Acid
- Procedures performed: Biopsy of bone, other bones; Other myectomy; Transfusion of packed cells; Transfusion of other serum; Venous catheterization, not elsewhere classified; Non-invasive mechanical ventilation
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 7461,
  "diagnoses_codes_current": "0389;70703;73008;5990;2639;99591;25000;3558;3441;78057;78791;7245;V4581;412;27801",
  "procedures_codes_current": "7749;8345;9904;9907;3893;9390",
  "medications_current": "Daptomycin;Alteplase (Catheter Clearance);Tigecycline;OxycoDONE (Immediate Release);Acetaminophen;Insulin;Vancomycin;Docusate Sodium;Senna;Iso-Osmotic Dextrose;Nicotine Patch;Pantoprazole;Simvastatin;Ipratropium Bromide MDI;Losartan Potassium;Heparin;Furosemide;Fentanyl Patch;Aspirin EC;Albuterol;Sodium Chloride 0.9%  Flush;1/2 NS;NS (Mini Bag Plus);Meropenem;Metoprolol;Nystatin Oral Suspension;Oxycodone-Acetaminophen;D5 1/2NS;Calcium Gluconate;NS;Dakins 1/4 Strength;LR;Nafcillin;Piperacillin-Tazobactam Na;Docusate Sodium (Liquid);MetRONIDAZOLE (FLagyl);Potassium Chloride;Enoxaparin Sodium;D5W;Phenylephrine;Aspirin;Phytonadione;Heparin Flush CVL  (100 units/ml);Famotidine;Bisacodyl;DiphenhydrAMINE;Zinc Sulfate;Multivitamins W/minerals;Ascorbic Acid",
  "label_lipid_next": 1,
  "label_diabetes_current": 1,
  "label_hypertension_current": 0,
  "label_obesity_current": 1,
  "label_cardio_next": 1,
  "label_kidney_next": 0,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Unspecified septicemia; Pressure ulcer, lower back; Acute osteomyelitis, other specified sites; Urinary tract infection, site not specified; Unspecified protein-calorie malnutrition; Sepsis; Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; Mononeuritis of lower limb, unspecified; Paraplegia; Unspecified sleep apnea; Diarrhea; Backache, unspecified; Aortocoronary bypass status; Old myocardial infarction; Morbid obesity\n- Medications prescribed: Daptomycin; Alteplase (Catheter Clearance); Tigecycline; OxycoDONE (Immediate Release); Acetaminophen; Insulin; Vancomycin; Docusate Sodium; Senna; Iso-Osmotic Dextrose; Nicotine Patch; Pantoprazole; Simvastatin; Ipratropium Bromide MDI; Losartan Potassium; Heparin; Furosemide; Fentanyl Patch; Aspirin EC; Albuterol; Sodium Chloride 0.9%  Flush; 1/2 NS; NS (Mini Bag Plus); Meropenem; Metoprolol; Nystatin Oral Suspension; Oxycodone-Acetaminophen; D5 1/2NS; Calcium Gluconate; NS; Dakins 1/4 Strength; LR; Nafcillin; Piperacillin-Tazobactam Na; Docusate Sodium (Liquid); MetRONIDAZOLE (FLagyl); Potassium Chloride; Enoxaparin Sodium; D5W; Phenylephrine; Aspirin; Phytonadione; Heparin Flush CVL  (100 units/ml); Famotidine; Bisacodyl; DiphenhydrAMINE; Zinc Sulfate; Multivitamins W/minerals; Ascorbic Acid\n- Procedures performed: Biopsy of bone, other bones; Other myectomy; Transfusion of packed cells; Transfusion of other serum; Venous catheterization, not elsewhere classified; Non-invasive mechanical ventilation"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_021 — pair_id 8261

- **Tags:** cross_model_disagreement, cross_model_disagreement_high, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (4):** lipid, diabetes, hypertension, cardio
- **Ground truth (task → label):** `{'lipid': 1, 'diabetes': 0, 'hypertension': 0, 'obesity': 0, 'cardio': 1, 'kidney': 1, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['diabetes', 'hypertension', 'stroke']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.8}, 'diabetes': {'pred': 1, 'prob': 0.9}, 'hypertension': {'pred': 1, 'prob': 0.85}, 'obesity': {'pred': 0, 'prob': 0.1}, 'cardio': {'pred': 1, 'prob': 0.95}, 'kidney': {'pred': 1, 'prob': 0.75}, 'stroke': {'pred': 1, 'prob': 0.7}}`

**Reasoning:**

```
The patient has a history of cardiovascular issues, including subendocardial infarction and coronary atherosclerosis, which supports the prediction of lipid metabolism disorders and cardiovascular disease. The presence of insulin indicates current diabetes management. Hypertension is likely given the medications prescribed, including Lisinopril and Metoprolol. There is no explicit evidence of obesity in the record. Acute kidney failure suggests a risk for kidney disease in the next visit, and the overall cardiovascular status raises the likelihood of a stroke.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['lipid', 'cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 41071;42741;42820;9971;4271;9972;5849;5990;41401;4280;E8790;4423;496;7101;4430. Current-visit narrative (truncated): - Diagnoses made: Subendocardial infarction, initial episode of care; Ventricular fibrillation; Systolic heart failure, unspecified; Cardiac complications, not elsewhere classified; Paroxysmal ventricular tachycardia; Peripheral vascular complications, not elsewhere classified; Acute kidney failure, unspecified; Urinary tract infection, site not specified; Coronary atherosclerosis of native coronary artery; Congestive heart failure, unspecified; Cardiac catheterization as the cause of abnormal reaction of patient, or of later complication, without mention of misadventure at time of procedure; Aneurysm of artery of lower extremity; Chronic airway obstruction, not elsewhere classified; Systemic sclerosis; Raynaud's syndrome - Medications prescribed: Heparin; Potassium Chloride; Sodium Bicarbonate; NS; D5W; Heparin Sodium; Sulfameth/Trimethoprim DS; Hydrocortisone Oint 0.5%; Zolpidem Tar...

### Current-visit narrative

```
- Diagnoses made: Subendocardial infarction, initial episode of care; Ventricular fibrillation; Systolic heart failure, unspecified; Cardiac complications, not elsewhere classified; Paroxysmal ventricular tachycardia; Peripheral vascular complications, not elsewhere classified; Acute kidney failure, unspecified; Urinary tract infection, site not specified; Coronary atherosclerosis of native coronary artery; Congestive heart failure, unspecified; Cardiac catheterization as the cause of abnormal reaction of patient, or of later complication, without mention of misadventure at time of procedure; Aneurysm of artery of lower extremity; Chronic airway obstruction, not elsewhere classified; Systemic sclerosis; Raynaud's syndrome
- Medications prescribed: Heparin; Potassium Chloride; Sodium Bicarbonate; NS; D5W; Heparin Sodium; Sulfameth/Trimethoprim DS; Hydrocortisone Oint 0.5%; Zolpidem Tartrate; Acetylcysteine 20%; Lisinopril; Lorazepam; Atropine Sulfate; Senna; Oxycodone-Acetaminophen; Furosemide; traZODONE; Insulin Human Regular; Calcium Gluconate; Magnesium Sulfate; Morphine Sulfate; Metoclopramide; Sucralfate; Acetaminophen; Aspirin EC; Aspirin; Ranitidine; Docusate Sodium; Docusate Sodium (Liquid); Clopidogrel Bisulfate; Isosorbide Mononitrate (Extended Release); Hydroxychloroquine Sulfate; Atorvastatin; Influenza Virus Vaccine; Sodium Chloride 0.9%  Flush; Pantoprazole; Metoprolol; Magnesium Oxide; Allopurinol; HydrALAzine; Lidocaine; Vancomycin; Iso-Osmotic Dextrose; Ondansetron; Insulin; Milk of Magnesia; Bisacodyl; Simvastatin; Amiodarone; Syringe
- Procedures performed: Implantation or replacement of automatic cardioverter/defibrillator, total system [AICD]; Combined right and left heart cardiac catheterization; Single internal mammary-coronary artery bypass; Percutaneous transluminal coronary angioplasty [PTCA]; Other repair of aneurysm; Insertion of drug-eluting coronary artery stent(s); Insertion of one vascular stent; Procedure on single vessel; Coronary arteriography using two catheters; Insertion of temporary transvenous pacemaker system
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 8261,
  "diagnoses_codes_current": "41071;42741;42820;9971;4271;9972;5849;5990;41401;4280;E8790;4423;496;7101;4430",
  "procedures_codes_current": "3794;3723;3615;66;3952;3607;45;40;8856;3778",
  "medications_current": "Heparin;Potassium Chloride;Sodium Bicarbonate;NS;D5W;Heparin Sodium;Sulfameth/Trimethoprim DS;Hydrocortisone Oint 0.5%;Zolpidem Tartrate;Acetylcysteine 20%;Lisinopril;Lorazepam;Atropine Sulfate;Senna;Oxycodone-Acetaminophen;Furosemide;traZODONE;Insulin Human Regular;Calcium Gluconate;Magnesium Sulfate;Morphine Sulfate;Metoclopramide;Sucralfate;Acetaminophen;Aspirin EC;Aspirin;Ranitidine;Docusate Sodium;Docusate Sodium (Liquid);Clopidogrel Bisulfate;Isosorbide Mononitrate (Extended Release);Hydroxychloroquine Sulfate;Atorvastatin;Influenza Virus Vaccine;Sodium Chloride 0.9%  Flush;Pantoprazole;Metoprolol;Magnesium Oxide;Allopurinol;HydrALAzine;Lidocaine;Vancomycin;Iso-Osmotic Dextrose;Ondansetron;Insulin;Milk of Magnesia;Bisacodyl;Simvastatin;Amiodarone;Syringe;Chlorhexidine Gluconate 0.12% Oral Rinse;Vial;Heparin Flush CVL  (100 units/ml);Pantoprazole Sodium;SW;Phenylephrine;Metolazone;Midazolam;LR;Dextrose 50%;Propofol;Nitroglycerin;Glycopyrrolate;Neostigmine;1/2 NS;Epinephrine",
  "label_lipid_next": 1,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 1,
  "label_kidney_next": 1,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Subendocardial infarction, initial episode of care; Ventricular fibrillation; Systolic heart failure, unspecified; Cardiac complications, not elsewhere classified; Paroxysmal ventricular tachycardia; Peripheral vascular complications, not elsewhere classified; Acute kidney failure, unspecified; Urinary tract infection, site not specified; Coronary atherosclerosis of native coronary artery; Congestive heart failure, unspecified; Cardiac catheterization as the cause of abnormal reaction of patient, or of later complication, without mention of misadventure at time of procedure; Aneurysm of artery of lower extremity; Chronic airway obstruction, not elsewhere classified; Systemic sclerosis; Raynaud's syndrome\n- Medications prescribed: Heparin; Potassium Chloride; Sodium Bicarbonate; NS; D5W; Heparin Sodium; Sulfameth/Trimethoprim DS; Hydrocortisone Oint 0.5%; Zolpidem Tartrate; Acetylcysteine 20%; Lisinopril; Lorazepam; Atropine Sulfate; Senna; Oxycodone-Acetaminophen; Furosemide; traZODONE; Insulin Human Regular; Calcium Gluconate; Magnesium Sulfate; Morphine Sulfate; Metoclopramide; Sucralfate; Acetaminophen; Aspirin EC; Aspirin; Ranitidine; Docusate Sodium; Docusate Sodium (Liquid); Clopidogrel Bisulfate; Isosorbide Mononitrate (Extended Release); Hydroxychloroquine Sulfate; Atorvastatin; Influenza Virus Vaccine; Sodium Chloride 0.9%  Flush; Pantoprazole; Metoprolol; Magnesium Oxide; Allopurinol; HydrALAzine; Lidocaine; Vancomycin; Iso-Osmotic Dextrose; Ondansetron; Insulin; Milk of Magnesia; Bisacodyl; Simvastatin; Amiodarone; Syringe\n- Procedures performed: Implantation or replacement of automatic cardioverter/defibrillator, total system [AICD]; Combined right and left heart cardiac catheterization; Single internal mammary-coronary artery bypass; Percutaneous transluminal coronary angioplasty [PTCA]; Other repair of aneurysm; Insertion of drug-eluting coronary artery stent(s); Insertion of one vascular stent; Procedure on single vessel; Coronary arteriography using two catheters; Insertion of temporary transvenous pacemaker system"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_022 — pair_id 8270

- **Tags:** cross_model_disagreement, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (2):** lipid, hypertension
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 1, 'hypertension': 0, 'obesity': 0, 'cardio': 0, 'kidney': 0, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['hypertension', 'cardio', 'stroke']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 1, 'prob': 1.0}, 'hypertension': {'pred': 1, 'prob': 0.75}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.75}, 'kidney': {'pred': 0, 'prob': 0.0}, 'stroke': {'pred': 1, 'prob': 0.75}}`

**Reasoning:**

```
The patient has a documented diagnosis of diabetes mellitus, which confirms current diabetes. Hypertension is likely given the prescription of Lisinopril, a common antihypertensive, but not explicitly stated as a diagnosis. There is no evidence of obesity in the record. The patient has multiple cardiovascular issues, including atrial fibrillation and cerebral artery occlusion, suggesting a high likelihood of future cardiovascular disease and stroke. There is no indication of kidney disease in the current visit.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['lipid', 'cardio']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.95}, 'diabetes': {'pred': 1, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.95}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 5770;51881;57451;43491;5990;2639;5119;42731;25000;2859. Current-visit narrative (truncated): - Diagnoses made: Acute pancreatitis; Acute respiratory failure; Calculus of bile duct without mention of cholecystitis, with obstruction; Cerebral artery occlusion, unspecified with cerebral infarction; Urinary tract infection, site not specified; Unspecified protein-calorie malnutrition; Unspecified pleural effusion; Atrial fibrillation; Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; Anemia, unspecified - Medications prescribed: PNEUMOcoccal Vac Polyvalent; Ipratropium Bromide Neb; Nystatin Ointment; Insulin; Warfarin; Furosemide; Metoprolol; Fluconazole; Sulfameth/Trimethoprim DS; D5W; Heparin Sodium; Nystatin Oral Suspension; Magnesium Sulfate; Famotidine; Nystatin; Potassium Chloride; SW; Piperacillin-Tazobactam Na; Vancomycin; Iso-Osmotic Dextrose; Diltiazem; Spironolactone; Sodium Chloride 0.9% Flush; NS; HYDROmorphon...

### Current-visit narrative

```
- Diagnoses made: Acute pancreatitis; Acute respiratory failure; Calculus of bile duct without mention of cholecystitis, with obstruction; Cerebral artery occlusion, unspecified with cerebral infarction; Urinary tract infection, site not specified; Unspecified protein-calorie malnutrition; Unspecified pleural effusion; Atrial fibrillation; Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; Anemia, unspecified
- Medications prescribed: PNEUMOcoccal Vac Polyvalent; Ipratropium Bromide Neb; Nystatin Ointment; Insulin; Warfarin; Furosemide; Metoprolol; Fluconazole; Sulfameth/Trimethoprim DS; D5W; Heparin Sodium; Nystatin Oral Suspension; Magnesium Sulfate; Famotidine; Nystatin; Potassium Chloride; SW; Piperacillin-Tazobactam Na; Vancomycin; Iso-Osmotic Dextrose; Diltiazem; Spironolactone; Sodium Chloride 0.9%  Flush; NS; HYDROmorphone (Dilaudid); Nystatin Cream; AcetaZOLamide Sodium; Albumin 5% (12.5g / 250mL); Maalox/Diphenhydramine/Lidocaine; Oxybutynin; Chlorhexidine Gluconate 0.12% Oral Rinse; LR; Fentanyl Citrate; Propofol; NS (Mini Bag Plus); Ampicillin-Sulbactam; Pantoprazole Sodium; Vial; HydrALAzine; Calcium Gluconate; Heparin; Influenza Virus Vaccine; Lisinopril; Latanoprost 0.005% Ophth. Soln.
- Procedures performed: Endoscopic insertion of stent (tube) into bile duct; Continuous invasive mechanical ventilation for 96 consecutive hours or more; Venous catheterization, not elsewhere classified; Enteral infusion of concentrated nutritional substances; Thoracentesis
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 8270,
  "diagnoses_codes_current": "5770;51881;57451;43491;5990;2639;5119;42731;25000;2859",
  "procedures_codes_current": "5187;9672;3893;966;3491",
  "medications_current": "PNEUMOcoccal Vac Polyvalent;Ipratropium Bromide Neb;Nystatin Ointment;Insulin;Warfarin;Furosemide;Metoprolol;Fluconazole;Sulfameth/Trimethoprim DS;D5W;Heparin Sodium;Nystatin Oral Suspension;Magnesium Sulfate;Famotidine;Nystatin;Potassium Chloride;SW;Piperacillin-Tazobactam Na;Vancomycin;Iso-Osmotic Dextrose;Diltiazem;Spironolactone;Sodium Chloride 0.9%  Flush;NS;HYDROmorphone (Dilaudid);Nystatin Cream;AcetaZOLamide Sodium;Albumin 5% (12.5g / 250mL);Maalox/Diphenhydramine/Lidocaine;Oxybutynin;Chlorhexidine Gluconate 0.12% Oral Rinse;LR;Fentanyl Citrate;Propofol;NS (Mini Bag Plus);Ampicillin-Sulbactam;Pantoprazole Sodium;Vial;HydrALAzine;Calcium Gluconate;Heparin;Influenza Virus Vaccine;Lisinopril;Latanoprost 0.005% Ophth. Soln.",
  "label_lipid_next": 0,
  "label_diabetes_current": 1,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 0,
  "label_kidney_next": 0,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Acute pancreatitis; Acute respiratory failure; Calculus of bile duct without mention of cholecystitis, with obstruction; Cerebral artery occlusion, unspecified with cerebral infarction; Urinary tract infection, site not specified; Unspecified protein-calorie malnutrition; Unspecified pleural effusion; Atrial fibrillation; Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; Anemia, unspecified\n- Medications prescribed: PNEUMOcoccal Vac Polyvalent; Ipratropium Bromide Neb; Nystatin Ointment; Insulin; Warfarin; Furosemide; Metoprolol; Fluconazole; Sulfameth/Trimethoprim DS; D5W; Heparin Sodium; Nystatin Oral Suspension; Magnesium Sulfate; Famotidine; Nystatin; Potassium Chloride; SW; Piperacillin-Tazobactam Na; Vancomycin; Iso-Osmotic Dextrose; Diltiazem; Spironolactone; Sodium Chloride 0.9%  Flush; NS; HYDROmorphone (Dilaudid); Nystatin Cream; AcetaZOLamide Sodium; Albumin 5% (12.5g / 250mL); Maalox/Diphenhydramine/Lidocaine; Oxybutynin; Chlorhexidine Gluconate 0.12% Oral Rinse; LR; Fentanyl Citrate; Propofol; NS (Mini Bag Plus); Ampicillin-Sulbactam; Pantoprazole Sodium; Vial; HydrALAzine; Calcium Gluconate; Heparin; Influenza Virus Vaccine; Lisinopril; Latanoprost 0.005% Ophth. Soln.\n- Procedures performed: Endoscopic insertion of stent (tube) into bile duct; Continuous invasive mechanical ventilation for 96 consecutive hours or more; Venous catheterization, not elsewhere classified; Enteral infusion of concentrated nutritional substances; Thoracentesis"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_023 — pair_id 10306

- **Tags:** cross_model_disagreement, gemma_parse_salvage, gemma_partial_output, high_probability_gap_disagreement
- **Disagreeing tasks (2):** diabetes, cardio
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 0, 'obesity': 0, 'cardio': 0, 'kidney': 0, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_json` (salvage=False, n_tasks_salvaged=0)
- Wrong vs gold: ['cardio', 'stroke']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 1, 'prob': 0.5}, 'kidney': {'pred': 0, 'prob': 0.0}, 'stroke': {'pred': 1, 'prob': 0.5}}`

**Reasoning:**

```
There is no evidence of lipid disorders, diabetes, hypertension, or obesity in the current visit record. However, the patient has a history of cerebral artery occlusion, which suggests a risk for cardiovascular disease and stroke in the next visit.
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['diabetes']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 1, 'prob': 1.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 2273;5070;2535;25541;7907;2536;99931;2251;V5867;2537;E8786;2530;37749;43490;73689;7836;32723;32727;3051;2449;2572;2809;0. Current-visit narrative (truncated): - Diagnoses made: Benign neoplasm of pituitary gland and craniopharyngeal duct; Pneumonitis due to inhalation of food or vomitus; Diabetes insipidus; Glucocorticoid deficiency; Bacteremia; Other disorders of neurohypophysis; Other and unspecified infection due to central venous catheter; Benign neoplasm of cranial nerves; Long-term (current) use of insulin; Iatrogenic pituitary disorders; Removal of other organ (partial) (total) causing abnormal patient reaction, or later complication, without mention of misadventure at time of operation; Acromegaly and gigantism; Other disorders of optic nerve; Cerebral artery occlusion, unspecified without mention of cerebral infarction; Other acquired deformity of other parts of limb; Polyphagia; Obstructive sleep apnea (adult)(pediatric); Central sleep apnea in conditions classified elsewhere; Tobacco use disorder; Unspecified acquired hypothyroid...

### Current-visit narrative

```
- Diagnoses made: Benign neoplasm of pituitary gland and craniopharyngeal duct; Pneumonitis due to inhalation of food or vomitus; Diabetes insipidus; Glucocorticoid deficiency; Bacteremia; Other disorders of neurohypophysis; Other and unspecified infection due to central venous catheter; Benign neoplasm of cranial nerves; Long-term (current) use of insulin; Iatrogenic pituitary disorders; Removal of other organ (partial) (total) causing abnormal patient reaction, or later complication, without mention of misadventure at time of operation; Acromegaly and gigantism; Other disorders of optic nerve; Cerebral artery occlusion, unspecified without mention of cerebral infarction; Other acquired deformity of other parts of limb; Polyphagia; Obstructive sleep apnea (adult)(pediatric); Central sleep apnea in conditions classified elsewhere; Tobacco use disorder; Unspecified acquired hypothyroidism; Other testicular hypofunction; Iron deficiency anemia, unspecified; Staphylococcus infection in conditions classified elsewhere and of unspecified site, other staphylococcus; Phlebitis and thrombophlebitis of superficial veins of upper extremities; Elevated blood pressure reading without diagnosis of hypertension
- Medications prescribed: DiphenhydrAMINE; Desmopressin Acetate; 5% Dextrose; Vancomycin; Insulin; Glucagon; Dextrose 50%; Hydrocortisone; Senna; MetFORMIN (Glucophage); Levothyroxine Sodium; Heparin Flush (10 units/ml); SW; Potassium Chloride; Metoprolol Tartrate; Nicotine Patch; Artificial Tear Ointment; Artificial Tears; Iso-Osmotic Dextrose; 1/2 NS; Ibuprofen; Ferrous Sulfate; Alteplase (Catheter Clearance); 0.9% Sodium Chloride; Syringe (0.9% Sodium Chloride); Desmopressin Nasal; Sodium Chloride 0.9%  Flush; Syringe; LeVETiracetam; Famotidine; Heparin; Dexamethasone; Soln; Chlorhexidine Gluconate 0.12% Oral Rinse; D5W; Fentanyl Citrate; Piperacillin-Tazobactam; D5 1/2NS; 0.9% Sodium Chloride (Mini Bag Plus); Insulin Human Regular; HydrALAzine; Iso-Osmotic Sodium Chloride; Magnesium Sulfate; HYDROmorphone (Dilaudid); LR; Acetaminophen; Potassium Chl 20 mEq / 1000 mL NS; Morphine Sulfate; Ondansetron; Acetaminophen w/Codeine
- Procedures performed: Partial excision of pituitary gland, transfrontal approach; Formation of cranial bone flap; Other excision or avulsion of cranial and peripheral nerves; Frontal sinusectomy; Venous catheterization, not elsewhere classified
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 10306,
  "diagnoses_codes_current": "2273;5070;2535;25541;7907;2536;99931;2251;V5867;2537;E8786;2530;37749;43490;73689;7836;32723;32727;3051;2449;2572;2809;04119;45182;7962",
  "procedures_codes_current": "761;203;407;2242;3893",
  "medications_current": "DiphenhydrAMINE;Desmopressin Acetate;5% Dextrose;Vancomycin;Insulin;Glucagon;Dextrose 50%;Hydrocortisone;Senna;MetFORMIN (Glucophage);Levothyroxine Sodium;Heparin Flush (10 units/ml);SW;Potassium Chloride;Metoprolol Tartrate;Nicotine Patch;Artificial Tear Ointment;Artificial Tears;Iso-Osmotic Dextrose;1/2 NS;Ibuprofen;Ferrous Sulfate;Alteplase (Catheter Clearance);0.9% Sodium Chloride;Syringe (0.9% Sodium Chloride);Desmopressin Nasal;Sodium Chloride 0.9%  Flush;Syringe;LeVETiracetam;Famotidine;Heparin;Dexamethasone;Soln;Chlorhexidine Gluconate 0.12% Oral Rinse;D5W;Fentanyl Citrate;Piperacillin-Tazobactam;D5 1/2NS;0.9% Sodium Chloride (Mini Bag Plus);Insulin Human Regular;HydrALAzine;Iso-Osmotic Sodium Chloride;Magnesium Sulfate;HYDROmorphone (Dilaudid);LR;Acetaminophen;Potassium Chl 20 mEq / 1000 mL NS;Morphine Sulfate;Ondansetron;Acetaminophen w/Codeine;Vial;Nitroglycerin;Cisatracurium Besylate;NiCARdipine IV;Pantoprazole Sodium;Propofol;Lorazepam;Bisacodyl;Docusate Sodium",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 0,
  "label_kidney_next": 0,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Benign neoplasm of pituitary gland and craniopharyngeal duct; Pneumonitis due to inhalation of food or vomitus; Diabetes insipidus; Glucocorticoid deficiency; Bacteremia; Other disorders of neurohypophysis; Other and unspecified infection due to central venous catheter; Benign neoplasm of cranial nerves; Long-term (current) use of insulin; Iatrogenic pituitary disorders; Removal of other organ (partial) (total) causing abnormal patient reaction, or later complication, without mention of misadventure at time of operation; Acromegaly and gigantism; Other disorders of optic nerve; Cerebral artery occlusion, unspecified without mention of cerebral infarction; Other acquired deformity of other parts of limb; Polyphagia; Obstructive sleep apnea (adult)(pediatric); Central sleep apnea in conditions classified elsewhere; Tobacco use disorder; Unspecified acquired hypothyroidism; Other testicular hypofunction; Iron deficiency anemia, unspecified; Staphylococcus infection in conditions classified elsewhere and of unspecified site, other staphylococcus; Phlebitis and thrombophlebitis of superficial veins of upper extremities; Elevated blood pressure reading without diagnosis of hypertension\n- Medications prescribed: DiphenhydrAMINE; Desmopressin Acetate; 5% Dextrose; Vancomycin; Insulin; Glucagon; Dextrose 50%; Hydrocortisone; Senna; MetFORMIN (Glucophage); Levothyroxine Sodium; Heparin Flush (10 units/ml); SW; Potassium Chloride; Metoprolol Tartrate; Nicotine Patch; Artificial Tear Ointment; Artificial Tears; Iso-Osmotic Dextrose; 1/2 NS; Ibuprofen; Ferrous Sulfate; Alteplase (Catheter Clearance); 0.9% Sodium Chloride; Syringe (0.9% Sodium Chloride); Desmopressin Nasal; Sodium Chloride 0.9%  Flush; Syringe; LeVETiracetam; Famotidine; Heparin; Dexamethasone; Soln; Chlorhexidine Gluconate 0.12% Oral Rinse; D5W; Fentanyl Citrate; Piperacillin-Tazobactam; D5 1/2NS; 0.9% Sodium Chloride (Mini Bag Plus); Insulin Human Regular; HydrALAzine; Iso-Osmotic Sodium Chloride; Magnesium Sulfate; HYDROmorphone (Dilaudid); LR; Acetaminophen; Potassium Chl 20 mEq / 1000 mL NS; Morphine Sulfate; Ondansetron; Acetaminophen w/Codeine\n- Procedures performed: Partial excision of pituitary gland, transfrontal approach; Formation of cranial bone flap; Other excision or avulsion of cranial and peripheral nerves; Frontal sinusectomy; Venous catheterization, not elsewhere classified"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?

## mt_expert_024 — pair_id 11638

- **Tags:** cross_model_disagreement, cross_model_disagreement_high, gemma_parse_salvage, gemma_partial_output, gpt_parse_salvage, high_probability_gap_disagreement, lipid_task_disagreement
- **Disagreeing tasks (4):** lipid, diabetes, hypertension, cardio
- **Ground truth (task → label):** `{'lipid': 0, 'diabetes': 0, 'hypertension': 0, 'obesity': 0, 'cardio': 1, 'kidney': 0, 'stroke': 0}`

### GPT-4o-mini

- Parser: `multitask_salvaged_full` (salvage=True, n_tasks_salvaged=7)
- Wrong vs gold: ['lipid', 'diabetes', 'hypertension', 'kidney', 'stroke']
- Per-task: `{'lipid': {'pred': 1, 'prob': 0.8}, 'diabetes': {'pred': 1, 'prob': 0.9}, 'hypertension': {'pred': 1, 'prob': 0.85}, 'obesity': {'pred': 0, 'prob': 0.1}, 'cardio': {'pred': 1, 'prob': 0.95}, 'kidney': {'pred': 1, 'prob': 0.75}, 'stroke': {'pred': 1, 'prob': 0.7}}`

**Reasoning:**

```
nan
```

### Gemma4

- Parser: `multitask_salvaged_partial` (salvage=True, n_tasks_salvaged=5)
- Wrong vs gold: ['cardio']
- Per-task: `{'lipid': {'pred': 0, 'prob': 0.0}, 'diabetes': {'pred': 0, 'prob': 0.0}, 'hypertension': {'pred': 0, 'prob': 0.0}, 'obesity': {'pred': 0, 'prob': 0.0}, 'cardio': {'pred': 0, 'prob': 0.0}, 'kidney': {'pred': None, 'prob': None}, 'stroke': {'pred': None, 'prob': None}}`

**Reasoning:**

```

```

### Clinical summary

De-identified cohort record (pair_id only). Diagnosis codes (truncated): 41071;2761;5990;41401;53081;311;3051. Current-visit narrative (truncated): - Diagnoses made: Subendocardial infarction, initial episode of care; Hyposmolality and/or hyponatremia; Urinary tract infection, site not specified; Coronary atherosclerosis of native coronary artery; Esophageal reflux; Depressive disorder, not elsewhere classified; Tobacco use disorder - Medications prescribed: Albuterol-Ipratropium; Ketorolac; HYDROmorphone (Dilaudid); Oxycodone-Acetaminophen; Metoprolol Tartrate; Furosemide; Iso-Osmotic Dextrose; Vancomycin; Lorazepam; Sodium Chloride 0.9% Flush; Cyclobenzaprine; Paroxetine; Albuterol 0.083% Neb Soln; Docusate Sodium; Aspirin EC; Insulin; Milk of Magnesia; Acetaminophen; Ipratropium Bromide Neb; Atorvastatin; Omeprazole; Ciprofloxacin HCl; SW; Magnesium Sulfate; Ibuprofen; Neostigmine; Docusate Sodium (Liquid); Nitroglycerin; Phenylephrine; Potassium Chloride; Insulin Human Regular; Calcium Gluconate; Morphine Sulfate; 5% Dextrose...

### Current-visit narrative

```
- Diagnoses made: Subendocardial infarction, initial episode of care; Hyposmolality and/or hyponatremia; Urinary tract infection, site not specified; Coronary atherosclerosis of native coronary artery; Esophageal reflux; Depressive disorder, not elsewhere classified; Tobacco use disorder
- Medications prescribed: Albuterol-Ipratropium; Ketorolac; HYDROmorphone (Dilaudid); Oxycodone-Acetaminophen; Metoprolol Tartrate; Furosemide; Iso-Osmotic Dextrose; Vancomycin; Lorazepam; Sodium Chloride 0.9%  Flush; Cyclobenzaprine; Paroxetine; Albuterol 0.083% Neb Soln; Docusate Sodium; Aspirin EC; Insulin; Milk of Magnesia; Acetaminophen; Ipratropium Bromide Neb; Atorvastatin; Omeprazole; Ciprofloxacin HCl; SW; Magnesium Sulfate; Ibuprofen; Neostigmine; Docusate Sodium (Liquid); Nitroglycerin; Phenylephrine; Potassium Chloride; Insulin Human Regular; Calcium Gluconate; Morphine Sulfate; 5% Dextrose; D5W; Syringe; 0.9% Sodium Chloride; Metoclopramide; Dextrose 50%; Aspirin; Nicotine Patch; Influenza Virus Vaccine; Heparin Sodium; Clopidogrel; Eptifibatide; Heparin; Nitroglycerin SL; Amiodarone; Chlorhexidine Gluconate 0.12% Oral Rinse; Meperidine
- Procedures performed: (Aorto)coronary bypass of two coronary arteries; Left heart cardiac catheterization; Single internal mammary-coronary artery bypass; Extracorporeal circulation auxiliary to open heart surgery; Coronary arteriography using two catheters
```

### Raw data example (exact source row excerpt)

```json
{
  "pair_id": 11638,
  "diagnoses_codes_current": "41071;2761;5990;41401;53081;311;3051",
  "procedures_codes_current": "3612;3722;3615;3961;8856",
  "medications_current": "Albuterol-Ipratropium;Ketorolac;HYDROmorphone (Dilaudid);Oxycodone-Acetaminophen;Metoprolol Tartrate;Furosemide;Iso-Osmotic Dextrose;Vancomycin;Lorazepam;Sodium Chloride 0.9%  Flush;Cyclobenzaprine;Paroxetine;Albuterol 0.083% Neb Soln;Docusate Sodium;Aspirin EC;Insulin;Milk of Magnesia;Acetaminophen;Ipratropium Bromide Neb;Atorvastatin;Omeprazole;Ciprofloxacin HCl;SW;Magnesium Sulfate;Ibuprofen;Neostigmine;Docusate Sodium (Liquid);Nitroglycerin;Phenylephrine;Potassium Chloride;Insulin Human Regular;Calcium Gluconate;Morphine Sulfate;5% Dextrose;D5W;Syringe;0.9% Sodium Chloride;Metoclopramide;Dextrose 50%;Aspirin;Nicotine Patch;Influenza Virus Vaccine;Heparin Sodium;Clopidogrel;Eptifibatide;Heparin;Nitroglycerin SL;Amiodarone;Chlorhexidine Gluconate 0.12% Oral Rinse;Meperidine;LR;Glycopyrrolate;Propofol",
  "label_lipid_next": 0,
  "label_diabetes_current": 0,
  "label_hypertension_current": 0,
  "label_obesity_current": 0,
  "label_cardio_next": 1,
  "label_kidney_next": 0,
  "label_stroke_next": 0,
  "narrative_current": "- Diagnoses made: Subendocardial infarction, initial episode of care; Hyposmolality and/or hyponatremia; Urinary tract infection, site not specified; Coronary atherosclerosis of native coronary artery; Esophageal reflux; Depressive disorder, not elsewhere classified; Tobacco use disorder\n- Medications prescribed: Albuterol-Ipratropium; Ketorolac; HYDROmorphone (Dilaudid); Oxycodone-Acetaminophen; Metoprolol Tartrate; Furosemide; Iso-Osmotic Dextrose; Vancomycin; Lorazepam; Sodium Chloride 0.9%  Flush; Cyclobenzaprine; Paroxetine; Albuterol 0.083% Neb Soln; Docusate Sodium; Aspirin EC; Insulin; Milk of Magnesia; Acetaminophen; Ipratropium Bromide Neb; Atorvastatin; Omeprazole; Ciprofloxacin HCl; SW; Magnesium Sulfate; Ibuprofen; Neostigmine; Docusate Sodium (Liquid); Nitroglycerin; Phenylephrine; Potassium Chloride; Insulin Human Regular; Calcium Gluconate; Morphine Sulfate; 5% Dextrose; D5W; Syringe; 0.9% Sodium Chloride; Metoclopramide; Dextrose 50%; Aspirin; Nicotine Patch; Influenza Virus Vaccine; Heparin Sodium; Clopidogrel; Eptifibatide; Heparin; Nitroglycerin SL; Amiodarone; Chlorhexidine Gluconate 0.12% Oral Rinse; Meperidine\n- Procedures performed: (Aorto)coronary bypass of two coronary arteries; Left heart cardiac catheterization; Single internal mammary-coronary artery bypass; Extracorporeal circulation auxiliary to open heart surgery; Coronary arteriography using two catheters"
}
```

### Clinician prompts

- For each clinical task, does the model's probability align with the chart evidence?
- Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?
- Are any predictions driven by missing or truncated model output (parse/salvage)?
- Which errors would be most consequential if used for decision support?
