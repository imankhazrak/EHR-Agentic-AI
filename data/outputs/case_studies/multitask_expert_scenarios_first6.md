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
