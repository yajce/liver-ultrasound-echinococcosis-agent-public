---
name: echo-hydatid-v4
description: Apply structured liver-ultrasound observation with a generalized evidence-integration example and an explicit bias counterbalance.
---

# Echo Hydatid Skill v4

## Goal

- Review one liver ultrasound image in an observation-first order.
- Make the primary binary decision: `echinococcosis` or `negative`.
- Structure observation and evidence integration without replacing the baseline task or teaching a complete body of medical knowledge.
- Do not force a subtype when the image does not clearly support one.

## Step 1: Validate the image

Use `diagnosis = not_applicable` only when:

- The image is clearly not a liver ultrasound; or
- Technical quality is so poor that no meaningful visual assessment is possible.

Low contrast, atypical appearance, a subtle lesion, or uncertainty alone does not make the image invalid. Continue whenever meaningful assessment remains possible.

## Step 2: Observe systematically

Observe before deciding. Describe only what is visible:

1. Whether a focal lesion is present.
2. Whether it appears `cystic`, `solid`, or `mixed`.
3. The lesion boundary or wall.
4. Its internal architecture.
5. The appearance of the surrounding liver.

Keep observation separate from diagnosis. Do not infer beyond the supplied frame or turn this sequence into a disease-feature checklist.

## Step 3: Integrate evidence

Base the diagnosis on the overall visible morphology and evidence actually present.

Ask whether observations form a coherent pattern, considering supporting and conflicting evidence. One absent expected feature is not sufficient evidence against echinococcosis. Do not require a fixed number of findings.

Do not create a score, feature weights, or a threshold. Integrate evidence qualitatively and choose the best-supported diagnosis.

## Step 4: Avoid two opposite errors

- **Overdiagnosis:** Do not diagnose echinococcosis from a generic or simple cystic lesion alone.
- **Underdiagnosis:** Do not reject echinococcosis merely because one or more expected-looking features are not visible in a single static image.

Balance positive, negative, and uncertain observations before deciding from a static image.

## Worked reasoning example

This demonstrates evidence integration, not a reusable disease rule.

**Observation**

The image is an interpretable liver ultrasound. A focal cystic lesion is visible. It is not internally uniform: several organized components are present within heterogeneous material. Its boundary has a layered appearance. Record these observations before assigning a diagnosis.

**Evidence integration**

Do not decide from cystic appearance alone. Ask whether the internal organization and boundary support the same interpretation or are unrelated, nonspecific findings. Here they provide mutually supporting positive evidence. The conclusion comes from coherence, not one observation acting as an automatic trigger.

**Decision**

The combined visible morphology provides coherent positive support, so `echinococcosis` is favored. This creates neither a fixed template nor a subtype requirement.

## Counterbalance against example bias

This example demonstrates the reasoning process, not a default preference for a positive diagnosis.

For every new image:

- Do not assume that a cystic or complex lesion is echinococcosis.
- Use only findings actually visible in that image.
- If the overall evidence is nonspecific or better supports a non-echinococcal lesion, classify it as `negative`.
- Do not force the new case to resemble the example.

## Output constraints

- Use only the existing fields: `image_valid`, `diagnosis`, `confidence`, `subtype`, and `findings`.
- Do not add fields.
- `diagnosis` must be exactly `echinococcosis`, `negative`, or `not_applicable`.
- For an invalid image, use `image_valid = false`, `diagnosis = not_applicable`, `subtype = null`, and `confidence = null`.
- Set `subtype = null` unless a subtype is very clearly supported.
- Keep `findings` concise and grounded in visible observations.
- Treat `confidence` as self-reported confidence, not as a calibrated probability.

## Provenance note

This experimental skill extends v3 with a worked reasoning example. The example was abstracted from source-provided morphology descriptions in the development data and rewritten as a general reasoning pattern. It does not include an evaluation image, image identifier, or verbatim source caption.

The source development image that inspired this example must not be counted in a later v4 development-set evaluation.
