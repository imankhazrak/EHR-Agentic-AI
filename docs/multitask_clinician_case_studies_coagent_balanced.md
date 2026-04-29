# Multitask Clinician Case Studies (CoAgent, Balanced)

Clinician-facing case review for `mimiciii_llm_gpt4o_mini_promptv3_multitask_test` using balanced examples by task.

## Scope and selection intent
- Tasks: `lipid_next`, `diabetes_current`, `hypertension_current`, `obesity_current`, `cardio_next`, `kidney_next`, `stroke_next`.
- Target balancing: 1 `TP`, 1 `TN`, 1 `FP`, 1 `FN` per task (28 cases total).
- Confidence ranking rule: maximize `abs(probability - 0.5)` within each task/outcome bucket.

## Task confusion overview counts (balanced review set)
| Task | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| `lipid_next` | 1 | 1 | 1 | 1 |
| `diabetes_current` | 1 | 1 | 1 | 1 |
| `hypertension_current` | 1 | 1 | 1 | 1 |
| `obesity_current` | 1 | 1 | 1 | 1 |
| `cardio_next` | 1 | 1 | 1 | 1 |
| `kidney_next` | 1 | 1 | 1 | 1 |
| `stroke_next` | 1 | 1 | 1 | 1 |

## Cross-task clinician checklist
- Verify temporal framing (`*_current` vs `*_next`) before acting on probability outputs.
- Reconcile diagnosis/procedure/medication evidence with model rationale and threshold behavior.
- For `FP` predictions, check for prophylaxis/history coding that may inflate risk.
- For `FN` predictions, check hidden severity cues (multi-morbidity, recurrence, prior severe events).
- Review cardio-renal-metabolic-neuro overlap before final adjudication.

## Task: `lipid_next`

### TP case
- **True label:** Yes
- **Predicted label:** Yes
- **Probability:** 0.80
- **Clinical summary:** Hyperlipidemia-related signals and statin exposure suggest true positive future lipid disorder risk.
- **Diagnosis snippet:** `2720`, `41401`, `42731`
- **Procedure snippet:** `3521`, `3616`, `3961`
- **Medication snippet:** `Atorvastatin`, `Aspirin EC`, `Carvedilol`
- **Clinician prompts:**
  - Which lipid-specific findings support next-admission persistence?
  - Is this signal actionable now (medication optimization vs follow-up labs)?

### TN case
- **True label:** No
- **Predicted label:** No
- **Probability:** 0.00
- **Clinical summary:** No direct lipid-disorder coding or treatment pattern in the current encounter context.
- **Diagnosis snippet:** `2920`, `30401`, `30560`
- **Procedure snippet:** `N/A`
- **Medication snippet:** `Methadone`, `Diazepam`, `Thiamine`
- **Clinician prompts:**
  - Does chart review support low short-term lipid concern?
  - Are there missing outpatient records that could change this classification?

### FP case
- **True label:** No
- **Predicted label:** Yes
- **Probability:** 0.75
- **Clinical summary:** Model overcalls lipid risk from broad cardiometabolic context without sufficient target-label evidence.
- **Diagnosis snippet:** `4280`, `42731`, `4589`
- **Procedure snippet:** `4513`, `4443`, `4523`
- **Medication snippet:** `Atorvastatin`, `Warfarin`, `Carvedilol`
- **Clinician prompts:**
  - Is statin use preventive or treatment-confirmatory for the label definition?
  - Would ICD mapping rules classify this as label-negative despite treatment?

### FN case
- **True label:** Yes
- **Predicted label:** No
- **Probability:** 0.10
- **Clinical summary:** Under-recognition of lipid trajectory likely from sparse direct language in the current narrative.
- **Diagnosis snippet:** `41401`, `4019`, `2720`
- **Procedure snippet:** `8108`, `309`, `9604`
- **Medication snippet:** `Aspirin`, `Metoprolol`, `Insulin`
- **Clinician prompts:**
  - Which latent lipid indicators were likely missed?
  - Should this trigger targeted manual review in production?

## Task: `diabetes_current`

### TP case
- **True label:** Yes
- **Predicted label:** Yes
- **Probability:** 0.90
- **Clinical summary:** Insulin use and diabetic coding support correct positive classification.
- **Diagnosis snippet:** `25040`, `5856`, `40391`
- **Procedure snippet:** `3995`
- **Medication snippet:** `Insulin`, `Atorvastatin`, `Metoprolol`
- **Clinician prompts:**
  - Is diabetes active and clinically managed in this admission?
  - Does kidney/cardiac context alter glycemic management interpretation?

### TN case
- **True label:** No
- **Predicted label:** No
- **Probability:** 0.00
- **Clinical summary:** No convincing diabetes indicators in the present admission profile.
- **Diagnosis snippet:** `2920`, `2763`, `2967`
- **Procedure snippet:** `N/A`
- **Medication snippet:** `Methadone`, `Diazepam`, `Quetiapine`
- **Clinician prompts:**
  - Could transient hyperglycemia confound this classification?
  - Is there external evidence requiring override?

### FP case
- **True label:** No
- **Predicted label:** Yes
- **Probability:** 0.80
- **Clinical summary:** Diabetes overcall appears driven by medication context without stable diagnostic confirmation.
- **Diagnosis snippet:** `79902`, `20300`, `486`
- **Procedure snippet:** `3893`
- **Medication snippet:** `Insulin`, `PredniSONE`, `Levofloxacin`
- **Clinician prompts:**
  - Was insulin short-term/reactive rather than chronic diabetes treatment?
  - Should steroid-related glycemic excursions be separated from diagnosis?

### FN case
- **True label:** Yes
- **Predicted label:** No
- **Probability:** 0.00
- **Clinical summary:** Likely miss in diabetes detection despite broader chronic disease burden.
- **Diagnosis snippet:** `2720`, `4019`, `41401`
- **Procedure snippet:** `8108`, `9671`
- **Medication snippet:** `Aspirin`, `Metoprolol`, `Enalapril`
- **Clinician prompts:**
  - Which diabetes-specific evidence is absent from narrative but present structurally?
  - Should this case be included in post-hoc calibration review?

## Task: `hypertension_current`

### TP case
- **True label:** Yes
- **Predicted label:** Yes
- **Probability:** 1.00
- **Clinical summary:** Hypertensive CKD pattern and antihypertensive therapy align with positive prediction.
- **Diagnosis snippet:** `40391`, `5856`, `75313`
- **Procedure snippet:** `5554`, `5349`
- **Medication snippet:** `NIFEdipine CR`, `Metoprolol Tartrate`, `Midodrine`
- **Clinician prompts:** Confirm active hypertension vs historical coding; confirm treatment intensity consistency.

### TN case
- **True label:** No
- **Predicted label:** No
- **Probability:** 0.00
- **Clinical summary:** Behavioral-health-dominant encounter with no clear hypertension evidence.
- **Diagnosis snippet:** `30401`, `30500`, `3009`
- **Procedure snippet:** `N/A`
- **Medication snippet:** `Methadone`, `Diazepam`, `Ibuprofen`
- **Clinician prompts:** Check if any elevated BP episodes were uncoded; verify true TN status.

### FP case
- **True label:** No
- **Predicted label:** Yes
- **Probability:** 0.80
- **Clinical summary:** Model likely conflates cardiovascular risk context with active hypertension diagnosis.
- **Diagnosis snippet:** `7213`, `5185`, `41401`
- **Procedure snippet:** `9604`, `309`
- **Medication snippet:** `Aspirin`, `Metoprolol`, `Insulin`
- **Clinician prompts:** Distinguish CAD risk factors from active hypertensive disease labeling.

### FN case
- **True label:** Yes
- **Predicted label:** No
- **Probability:** 0.10
- **Clinical summary:** Potential under-detection where hypertension evidence is indirect or fragmented.
- **Diagnosis snippet:** `4280`, `41401`, `25000`
- **Procedure snippet:** `N/A`
- **Medication snippet:** `Enalapril`, `Carvedilol`, `Aspirin EC`
- **Clinician prompts:** What non-obvious indicators should trigger a positive hypertension call?

## Task: `obesity_current`

### TP case
- **True label:** Yes
- **Predicted label:** Yes
- **Probability:** 0.90
- **Clinical summary:** Obesity-linked cardiometabolic burden with explicit supportive context.
- **Diagnosis snippet:** `278xx`-compatible profile, `4019`, `2720`
- **Procedure snippet:** `N/A`
- **Medication snippet:** `Insulin`, `Amlodipine`, `Atorvastatin`
- **Clinician prompts:** Are obesity indicators explicit enough for reliable label capture?

### TN case
- **True label:** No
- **Predicted label:** No
- **Probability:** 0.00
- **Clinical summary:** No obesity-specific coding or treatment pattern in this encounter.
- **Diagnosis snippet:** `2920`, `07054`, `30411`
- **Procedure snippet:** `N/A`
- **Medication snippet:** `Methadone`, `Diazepam`, `FoLIC Acid`
- **Clinician prompts:** Could obesity be present but uncoded in this admission?

### FP case
- **True label:** No
- **Predicted label:** Yes
- **Probability:** 0.75
- **Clinical summary:** Positive obesity prediction appears to overgeneralize from metabolic comorbidity.
- **Diagnosis snippet:** `25040`, `40391`, `5856`
- **Procedure snippet:** `3995`
- **Medication snippet:** `Insulin`, `Sevelamer`, `Atorvastatin`
- **Clinician prompts:** Which features caused overcall despite no obesity target label?

### FN case
- **True label:** Yes
- **Predicted label:** No
- **Probability:** 0.10
- **Clinical summary:** Model misses obesity where documentation may be sparse or indirect.
- **Diagnosis snippet:** `486`, `4019`, `2720`
- **Procedure snippet:** `3893`
- **Medication snippet:** `Insulin`, `PredniSONE`, `Furosemide`
- **Clinician prompts:** Should anthropometric cues be made explicit in clinician-facing prompts?

## Task: `cardio_next`

### TP case
- **True label:** Yes
- **Predicted label:** Yes
- **Probability:** 0.85
- **Clinical summary:** Strong cardiovascular comorbidity and therapies support true positive next-admission risk.
- **Diagnosis snippet:** `41401`, `42789`, `4280`
- **Procedure snippet:** `3607`, `3722`, `3995`
- **Medication snippet:** `Nitroglycerin`, `Clopidogrel`, `Metoprolol`
- **Clinician prompts:** Is there a near-term preventive action implied by this risk profile?

### TN case
- **True label:** No
- **Predicted label:** No
- **Probability:** 0.00
- **Clinical summary:** No compelling cardiovascular trajectory markers in this case context.
- **Diagnosis snippet:** `29284`, `30401`, `07054`
- **Procedure snippet:** `N/A`
- **Medication snippet:** `Methadone`, `Quetiapine`, `Diazepam`
- **Clinician prompts:** Confirm absence of structural heart risk not captured in coded fields.

### FP case
- **True label:** No
- **Predicted label:** Yes
- **Probability:** 0.75
- **Clinical summary:** High-risk comorbidity likely drove overprediction despite label-negative next outcome.
- **Diagnosis snippet:** `40391`, `5856`, `34839`
- **Procedure snippet:** `N/A`
- **Medication snippet:** `Labetalol`, `NIFEdipine CR`, `Warfarin`
- **Clinician prompts:** Which non-cardiac factors may have triggered false positive generalization?

### FN case
- **True label:** Yes
- **Predicted label:** No
- **Probability:** 0.10
- **Clinical summary:** Potentially missed evolving cardiovascular risk in mixed clinical presentations.
- **Diagnosis snippet:** `2720`, `4019`, `V4581`
- **Procedure snippet:** `8108`, `9604`
- **Medication snippet:** `Aspirin`, `Metoprolol`, `Enalapril`
- **Clinician prompts:** Which latent signals should prompt override despite low model probability?

## Task: `kidney_next`

### TP case
- **True label:** Yes
- **Predicted label:** Yes
- **Probability:** 1.00
- **Clinical summary:** Renal failure/CKD profile with renal-focused therapy strongly supports true positive.
- **Diagnosis snippet:** `5856`, `40391`, `75313`
- **Procedure snippet:** `3995`, `5491`
- **Medication snippet:** `sevelamer`, `Nephrocaps`, `Calcium Carbonate`
- **Clinician prompts:** Is this persistent kidney disease vs acute-on-chronic exacerbation?

### TN case
- **True label:** No
- **Predicted label:** No
- **Probability:** 0.00
- **Clinical summary:** No clear renal progression indicators in current-to-next timeline.
- **Diagnosis snippet:** `2920`, `30560`, `3009`
- **Procedure snippet:** `N/A`
- **Medication snippet:** `Methadone`, `Ibuprofen`, `Diazepam`
- **Clinician prompts:** Any lab-only renal signal absent from coded data?

### FP case
- **True label:** No
- **Predicted label:** Yes
- **Probability:** 0.75
- **Clinical summary:** Model likely influenced by transient kidney-related context without durable next label.
- **Diagnosis snippet:** `78820`, `5990`, `73008`
- **Procedure snippet:** `9671`, `9604`
- **Medication snippet:** `Vancomycin`, `Heparin`, `Potassium Chloride`
- **Clinician prompts:** Was renal risk acute/transient rather than chronic next-admission disease?

### FN case
- **True label:** Yes
- **Predicted label:** No
- **Probability:** 0.20
- **Clinical summary:** Under-calling kidney risk where multi-system illness may obscure renal trajectory.
- **Diagnosis snippet:** `41401`, `42731`, `4019`
- **Procedure snippet:** `3521`, `3616`
- **Medication snippet:** `Furosemide`, `Aspirin`, `Atorvastatin`
- **Clinician prompts:** Which renal markers should trigger escalation despite negative prediction?

## Task: `stroke_next`

### TP case
- **True label:** Yes
- **Predicted label:** Yes
- **Probability:** 0.90
- **Clinical summary:** Prior intracranial pathology and risk profile align with true positive stroke risk.
- **Diagnosis snippet:** `431`, `486`, `4019`
- **Procedure snippet:** `966`, `9672`
- **Medication snippet:** `Aspirin`, `Levetiracetam`, `NiCARdipine IV`
- **Clinician prompts:** Is secondary prevention documented and adequate before discharge?

### TN case
- **True label:** No
- **Predicted label:** No
- **Probability:** 0.00
- **Clinical summary:** No neurologic vascular progression signal toward next-admission stroke.
- **Diagnosis snippet:** `2920`, `2763`, `30411`
- **Procedure snippet:** `N/A`
- **Medication snippet:** `Methadone`, `Diazepam`, `Ondansetron`
- **Clinician prompts:** Any missing neurology history that could alter this classification?

### FP case
- **True label:** No
- **Predicted label:** Yes
- **Probability:** 0.80
- **Clinical summary:** Model likely overweights vascular comorbidity in absence of true next stroke label.
- **Diagnosis snippet:** `40301`, `5856`, `27541`
- **Procedure snippet:** `N/A`
- **Medication snippet:** `Warfarin`, `Labetalol`, `HydrALAzine`
- **Clinician prompts:** Are anticoagulation and CKD signals causing stroke overprediction?

### FN case
- **True label:** Yes
- **Predicted label:** No
- **Probability:** 0.00
- **Clinical summary:** Missed stroke-risk evolution where precursor cerebrovascular cues may be subtle.
- **Diagnosis snippet:** `43411`, `431`, `4476`
- **Procedure snippet:** `331`, `3324`
- **Medication snippet:** `Metoprolol`, `Mannitol`, `Lorazepam`
- **Clinician prompts:** Which chart elements would support a manual stroke-risk override?

---
Data sources referenced:
- `data/outputs/mimiciii_llm_gpt4o_mini_promptv3_multitask_test/llm_coagent_results.csv`
- `data/processed/mimiciii_multitask/test.csv`
