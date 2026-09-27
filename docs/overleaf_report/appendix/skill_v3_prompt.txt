---
name: echo-hydatid-v3
description: Apply a minimal structured-observation workflow for experimental liver-ultrasound classification without an explicit disease-feature checklist.
---

# Echo Hydatid Skill v3

## Goal

- Review one liver ultrasound image in a consistent, observation-first order.
- Make the primary binary decision: `echinococcosis` or `negative`.
- Use this skill to structure image review, not to replace the baseline task or teach a complete body of medical knowledge.
- Do not force a subtype when the image does not clearly support one.

## Step 1: Validate the image

Use `diagnosis = not_applicable` only when:

- The image is clearly not a liver ultrasound; or
- Technical quality is so poor that no meaningful visual assessment is possible.

Do not mark the image invalid merely because contrast is low, the appearance is atypical, a lesion is subtle, or the interpretation is uncertain. If a meaningful assessment remains possible, continue with the binary task.

## Step 2: Observe systematically

Observe before deciding. Describe only what is visible:

1. Whether a focal lesion is present.
2. Whether the lesion appears `cystic`, `solid`, or `mixed`.
3. The lesion boundary or wall.
4. Its internal architecture.
5. Any visible internal echoes or structures.
6. The appearance of the surrounding liver.

Keep observations separate from the final diagnosis. Do not infer findings outside the supplied frame, and do not turn this sequence into a disease-feature checklist.

## Step 3: Integrate evidence

Base the diagnosis on the overall visible morphology. Use positive evidence that is actually present.

Do not treat the absence of one expected feature as sufficient evidence against echinococcosis. Do not require a fixed number of characteristic findings.

Do not create a score, assign feature weights, or invent a decision threshold. Integrate the visible evidence qualitatively, then choose the diagnosis that is best supported by the complete image.

## Step 4: Avoid two opposite errors

- **Error A — overdiagnosis:** Do not diagnose echinococcosis from a generic or simple cystic lesion alone.
- **Error B — underdiagnosis:** Do not reject echinococcosis merely because one or more classic-looking features are not visible in a single static image.

A single static ultrasound image may not show every characteristic feature. Balance positive and negative evidence before making the final decision.

## Output constraints

- Use only the existing fields: `image_valid`, `diagnosis`, `confidence`, `subtype`, and `findings`.
- Do not add fields.
- `diagnosis` must be exactly `echinococcosis`, `negative`, or `not_applicable`.
- When the image is invalid, use `image_valid = false`, `diagnosis = not_applicable`, `subtype = null`, and `confidence = null`.
- Set `subtype = null` unless a subtype is very clearly supported.
- Keep `findings` concise and grounded in visible observations.
- Treat `confidence` as self-reported confidence, not as a calibrated probability.

## Provenance note

This skill is an experimental structured-observation prompt. It intentionally avoids an explicit disease-feature checklist in order to test whether reasoning structure alone improves performance.
