---
name: echo-hydatid-v1
description: Apply a structured liver-ultrasound review workflow for experimental echinococcosis-versus-negative classification while preserving the existing output schema.
---

# Echo Hydatid Skill v1

## Goal

- Review one liver ultrasound image using a consistent, observation-first workflow.
- Make the primary binary decision: `echinococcosis` or `negative`.
- Use this skill only as an additional domain-specific assessment procedure. It does not replace the baseline task instructions, create a new classifier, or change the existing JSON output schema.
- Treat WHO stage as optional supporting information. Do not require a stage to make the binary decision.

## Step 1: Validate the image

Before interpreting pathology, confirm both of the following:

- The input is an interpretable liver ultrasound image.
- Liver parenchyma or a suspected focal lesion is sufficiently visible for assessment.

If the image is not an analyzable liver ultrasound, stop the medical assessment and return:

- `image_valid = false`
- `diagnosis = not_applicable`
- `subtype = null`
- `confidence = null`

Do not continue with diagnostic inference after declaring the image invalid.

## Step 2: Observe before diagnosing

Follow the rule: **observation first, diagnosis second**. Record only features that are visible in the supplied image. Review the image in this order:

1. Determine whether a focal lesion is present.
2. If present, describe it as `cystic`, `solid`, or `mixed`.
3. Assess whether the lesion is `unilocular` or `multivesicular`.
4. Describe the cyst wall, lesion boundary, or both.
5. Look for internal membranes.
6. Look for daughter cysts or daughter vesicles.
7. Assess internal echoes or heterogeneous material.
8. Look for calcification.
9. Look for posterior acoustic shadowing.

Do not infer features that are outside the field of view. Do not treat a possible observation as confirmed merely because it would fit a diagnosis.

## Step 3: Look for characteristic CE morphology

When assessing possible cystic echinococcosis (CE), give particular attention to the following morphology:

- Double-wall morphology.
- Hydatid sand or characteristic fine internal echoes.
- Daughter cysts.
- A multivesicular honeycomb or rosette-like appearance.
- A detached internal membrane or water-lily morphology.
- Daughter vesicles within a predominantly solid matrix.
- Characteristic degenerative ball-of-wool or canalicular morphology.
- A near-complete eggshell-calcified wall with posterior acoustic shadowing.

These findings increase support for CE, but no item in this list should automatically determine the diagnosis by itself. Integrate the complete visible morphology, the consistency of the findings, and the limitations of the single image.

## Step 4: Avoid overdiagnosis

A simple round or oval anechoic cyst, without characteristic CE morphology, is not sufficient by itself to diagnose echinococcosis.

Calcification alone is not sufficient evidence to determine CE diagnosis, stage, or viability.

Do not convert a nonspecific focal liver finding into an echinococcosis diagnosis solely because it is cystic. Distinguish characteristic CE morphology from findings that may occur in other liver lesions.

## Step 5: Integrate the evidence

Make the final binary decision from the overall pattern rather than from a single isolated feature:

- If characteristic CE morphology is clearly present, use `diagnosis = echinococcosis`.
- If multiple mutually consistent CE features are present, use `diagnosis = echinococcosis`.
- If only nonspecific cystic or focal liver findings are present and characteristic CE morphology is absent, do not infer echinococcosis from cystic morphology alone.
- Otherwise, use `diagnosis = negative`.

Keep this integration qualitative. Do not create a numeric score, assign feature weights, or invent decision thresholds. Express uncertainty in the existing `confidence` output, but do not describe self-reported confidence as a calibrated probability.

## Step 6: WHO stage

WHO stage is optional. Assign `CE1`, `CE2`, `CE3a`, `CE3b`, `CE4`, or `CE5` only when the visible morphology clearly supports that stage.

If the stage is not sufficiently clear, use `subtype = null`. Do not force a stage merely to produce a more specific answer.

A single static ultrasound image cannot reliably establish signs that require dynamic observation. Do not infer movement or other dynamic behavior from a static frame.

## Output constraints

- Continue to use the existing JSON schema and only these fields: `image_valid`, `diagnosis`, `confidence`, `subtype`, and `findings`.
- Do not add new output fields.
- `diagnosis` must be exactly one of: `echinococcosis`, `negative`, or `not_applicable`.
- When the image is invalid, use the values specified in Step 1.
- Use `subtype = null` whenever a WHO stage is not clearly supported.
- Keep `findings` focused on visible observations and make the diagnosis only after those observations have been integrated.
- Treat `confidence` as the model's self-reported certainty, not as a calibrated probability.

## References / provenance note

This skill is intended to summarize a structured ultrasound assessment workflow derived from established CE ultrasound morphology and WHO-IWGE style classification concepts. It should be treated as an experimental decision-support prompt, not as a clinical diagnostic guideline.
