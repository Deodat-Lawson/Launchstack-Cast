# DroneSearch + RescueRank
## A Multimodal Person Retrieval and Triage Engine over Aerial Video

**Course:** EN.601.466/666 — Information Retrieval and Web Agents (Spring 2026)
**Final Project Option:** Option 9 — *IR on Different Source Modalities (Video)*, with elements of Option 10 (RAG/LLM-mediated query)
**Author:** Timothy Lin (timothylinziqi@gmail.com)
**Section:** 466 / 666 *(to be confirmed)*

---

## 1. One-Paragraph Summary

DroneSearch treats every person detected in a drone's video stream as a *document* and builds a full information-retrieval system over that corpus: tokenization analogs (visual feature extraction and attribute tagging), an inverted index over discrete attributes, a dense vector index over CLIP embeddings, TF-IDF-style weighting, hybrid Boolean+ranked retrieval, Rocchio relevance feedback, and clustering for de-duplication of the same person across frames. On top of the retrieval core, a lightweight monitoring agent (RescueRank) fires alerts when a query target reappears or when a high-priority detection enters the field of view — the autonomous-decision-making layer. The long-term vision is a drone that can be told "find the missing hiker in the red jacket and tell me when you see them again" and act on it; this project builds the IR substrate that vision needs, evaluated rigorously on held-out aerial footage.

---

## 2. Why This Is Option 9, Not a Vision Project

The course rubric weighs *retrieval, classification, indexing, weighting, filtering, IE, and rigorous evaluation*. This proposal deliberately uses **off-the-shelf object detection (YOLOv8)** as a black-box tokenizer — the analog of using a pre-built sentence segmenter in a text-IR pipeline — and spends the project's innovation budget on the IR layer. Concretely:

| Course-rubric IR concept (from HW1–3 and lectures)              | Project component                                              |
|------------------------------------------------------------------|-----------------------------------------------------------------|
| Tokenization & segmentation (HW1)                                | Detector + tracker produces person crops and trajectory IDs    |
| Stopwords, region tags                                           | Filter low-confidence / occluded crops; tag head/torso/legs    |
| TF-IDF term weighting                                            | IDF-weighted attribute tags (rare colors/actions weighted up)  |
| Vector space model + centroids (HW2)                             | Per-identity centroid embedding; "friend/notfriend" analog     |
| Cosine similarity, ranked retrieval                              | CLIP embedding similarity + FAISS ranked retrieval             |
| Inverted index                                                   | Posting lists over discrete attribute tags                     |
| Boolean + ranked hybrid retrieval                                | Filter by attributes, rank by embedding score                  |
| Bayesian / decision tree classifier (HW3)                        | "Likely person in distress" triage classifier (RescueRank)     |
| Relevance feedback (Rocchio)                                     | Operator marks correct/incorrect matches → query vector update |
| Document clustering                                              | DBSCAN over embeddings to dedupe re-observations               |
| Information extraction / template filling                        | Per-person record: bbox, time, attributes, GPS-proxy, action   |
| Routing/filtering                                                | Live-stream agent routes detections to alert / log / discard   |
| Cross-modal/cross-lingual flavor (Option 6 echo)                 | Text query → image retrieval via CLIP (cross-modal)            |
| LLM/RAG flavor (Option 10 echo)                                  | Optional natural-language query parser + flight-report writer  |
| Empirical evaluation on held-out data                            | Manual query benchmark with P@k, MAP, NDCG; no posthoc grading |

This mapping is the spine of the writeup. Every grader-facing claim ("we implemented relevance feedback") points back to a concrete piece of code with a concrete number.

---

## 3. System Architecture

The system has four layers. Data flows top to bottom; agent decisions flow back up.

**Layer 1 — Ingestion & Detection.** Aerial video is ingested at 1 fps (sampled to keep CLIP cost tractable). YOLOv8-x detects person bounding boxes; ByteTrack maintains short-term tracks. Each track produces a stable `track_id`. Output: a stream of `(frame_id, t, bbox, track_id, det_conf)` records.

**Layer 2 — Feature & Attribute Extraction.** Each crop passes through (a) a CLIP image encoder for a 512-d dense embedding, and (b) a lightweight attribute tagger (clothing color via HSV histogram + zero-shot CLIP for tags like "running", "walking", "carrying backpack", "lying down"). The crop is segmented into head/torso/legs regions; region-weighted tags get boosted (analog of header/body region weighting in HW1's mail filter). Output: per-detection `Document` with `embedding`, `tags[]`, `region_weights`, and metadata.

**Layer 3 — Indexing & Retrieval.**
- *Dense index:* FAISS HNSW over per-detection CLIP embeddings.
- *Inverted index:* posting lists keyed on attribute tags, with per-document TF (1 since tags are binary) and corpus-level IDF.
- *Identity index:* per-`track_id` centroid embedding (mean of detections in track) — the direct centroid analog from HW2.
- *Query types supported:*
  1. **Image Re-ID:** "this crop → find again" (cosine over dense index).
  2. **Text → image:** "person in red jacket running" (CLIP text encoder → dense index).
  3. **Boolean+ranked hybrid:** filter `tags ⊇ {red_jacket, running}` then rank by dense similarity.
  4. **Identity query:** "where was track #42 last seen" (lookup over identity index).
- *Ranking score:* `score = α · cos(q, d) + β · sum_t IDF(t)·match(t)` with α, β tuned on a dev split.

**Layer 4 — Agent / Decision.**
- *Query agent:* receives operator query (image, text, or NL), executes retrieval, returns top-k with thumbnails.
- *Relevance feedback loop:* operator marks `relevant`/`not_relevant`; Rocchio updates query vector and re-ranks.
- *Watch / alert agent:* given a "standing query," monitors the live ingestion stream and fires an alert when a new detection scores above threshold τ. This is the autonomous-decision component, kept intentionally **light** — the focus stays on IR quality.
- *Triage classifier (RescueRank):* a Bayesian classifier (course rubric calls for an alternative to vector model) over hand-engineered features `{prone_pose, unusual_terrain, isolated, motionless_duration}` outputs `P(distress | features)`. Detections above threshold are escalated.

A tiny Streamlit UI exposes (a) image-upload Re-ID, (b) text-query search, (c) relevance-feedback buttons, (d) the live-alert log.

---

## 4. Data

**Primary:** [VisDrone](http://aiskyeye.com) — drone-perspective benchmark with detection, single-object tracking, and multi-object tracking annotations across hours of urban aerial footage. Academic license.

**Re-ID side data (optional):** A held-out subset of MARS or a small custom set of cross-camera person crops to validate the embedding-based identity matching.

**Search-and-rescue narrative dataset (optional, for RescueRank story):** [SARD](https://ieee-dataport.org/open-access/search-and-rescue-image-dataset-person-detection-sard) — drone images of people in wilderness search-and-rescue scenarios. Useful for the triage layer's distress feature engineering.

**Evaluation queries:** Manually constructed before any tuning (rubric requirement #6: no posthoc grading). 30 image queries + 30 text queries + 10 standing-query alert scenarios, with relevance judgments labeled by hand on held-out video segments.

**Data hygiene (rubric requirement on robot citizenship):** All datasets used under their academic licenses. No web crawling outside JHU; no live drone flights; no PII beyond what the public benchmarks already release.

---

## 5. Empirical Evaluation Plan

This is the section graders will read closely.

**Held-out splits.** Three VisDrone sequences not used for any tuning, IDF computation, or hyperparameter selection. Splits frozen at week 2.

**Metrics.**
- Image-query Re-ID: Precision@1, Precision@5, mAP across the 30-query benchmark.
- Text-query retrieval: Precision@5, Precision@10, NDCG@10.
- Hybrid vs dense-only ablation: same metrics, paired across queries.
- Relevance-feedback ablation: NDCG@10 before vs after one round of RF.
- Dedup clustering: pairwise precision/recall on a manually annotated identity-link set.
- Triage classifier: precision@k of "flagged" detections being true distress cases on SARD held-out frames.
- Alert agent: per-scenario time-to-first-alert vs ground-truth target reappearance time, plus false-alert rate.

**Baselines.**
1. Brute-force CLIP cosine similarity (no inverted index, no RF).
2. Tag-only Boolean retrieval (no embeddings).
3. Random ranking (sanity floor).

**Reporting.** All numbers in the final writeup with confidence intervals from bootstrap over query set. Screenshots per rubric item #5. Failure-case analysis per rubric item #4 (limitations).

---

## 6. Milestones (14-Week Semester)

| Wk | Phase                          | Deliverable                                          |
|----|--------------------------------|------------------------------------------------------|
| 1  | Setup                          | Repo skeleton, VisDrone downloaded, baseline detector running on one clip |
| 2  | Held-out splits frozen         | Train/dev/test split file checked into repo; **no further peeking at test** |
| 3  | Embeddings + dense index       | CLIP feature extraction pipeline; FAISS index over dev split |
| 4  | Image-query Re-ID baseline     | First MAP/P@k numbers on dev set                     |
| 5  | Attribute tagger + region wts  | Tag vocab, IDF table, region-weighted tagging        |
| 6  | Inverted index + hybrid score  | Boolean+ranked retrieval working                     |
| 7  | Text-query support (CLIP-text) | Cross-modal queries on dev set                       |
| 8  | Relevance feedback (Rocchio)   | RF loop with operator-marking UI                     |
| 9  | Dedup clustering               | DBSCAN over track centroids; identity-link metrics   |
| 10 | RescueRank Bayesian triage     | Distress classifier with hand-engineered features    |
| 11 | Alert agent                    | Standing-query watcher + alert log                   |
| 12 | Frozen system → run on test    | All metrics computed on the held-out set, **once**   |
| 13 | Writeup + screenshots          | First full draft of PDF report                       |
| 14 | Polish + submission            | Final PDF, code zip, demo video                      |

**Buffer rule.** If we slip by week 8, drop dedup clustering and RescueRank triage; keep retrieval + RF + alerts. The IR core must be done.

---

## 7. Risks and Mitigations

| Risk                                                          | Mitigation                                                |
|---------------------------------------------------------------|-----------------------------------------------------------|
| CLIP embedding cost over hours of video                       | Sample at 1 fps; cache embeddings to disk; only re-embed  |
| Drone-perspective Re-ID is genuinely hard (small bboxes, top-down) | Frame the project around "approximate retrieval"; document failure modes honestly in §limitations |
| Attribute tagger noisy on aerial crops                        | Use zero-shot CLIP rather than training a tagger; eval IDF on dev to filter tags that always fire |
| Posthoc grading temptation (rubric forbids it)                | Test split frozen at week 2 with a hash committed to git  |
| Scope creep (esp. AirSim demo)                                | AirSim is explicitly a *stretch* item, not a milestone    |

---

## 8. What Goes Beyond the Rubric (Selling Points)

These are the items I'll list under writeup section (3) "achievements / strengths":

1. **Cross-modal IR on a non-text modality** — text queries against an image corpus via CLIP. This is the heart of Option 9.
2. **End-to-end live-stream agent** — not just a static index; a watcher that fires on standing queries, demonstrating the "Web Agent" half of the course title applied to a non-web source.
3. **Two retrieval models compared** (vector-space cosine vs. hybrid Boolean+TF-IDF) per rubric expectation that the project not be a "simple bag-of-words vector comparison."
4. **Two classification approaches** (vector centroid for identity, Bayesian for triage) per the same rubric expectation.
5. **Relevance feedback** — Rocchio implemented end-to-end with a UI, with measurable NDCG lift.
6. **Honest evaluation** — frozen test split, bootstrap CIs, baselines including a sanity floor, ablation table.
7. **Clean stepping-stone toward real autonomy** — the live-stream interface is identical whether the source is a video file, AirSim, or a Tello drone, so the next semester's project is "swap the source."

---

## 9. Limitations (Honest, For The Final Writeup)

- No real flight; no real-time guarantees beyond what 1-fps sampling allows.
- Re-ID quality is bounded by detection-crop resolution, which is poor for aerial top-down views.
- Triage classifier uses hand-engineered features; not a learned end-to-end model.
- Standing-query alerts use a single threshold τ; no per-query calibration.

---

## 10. Tech Stack

Python 3.11; PyTorch + ultralytics (YOLOv8); open_clip; FAISS; scikit-learn (Bayesian + DBSCAN); Streamlit for UI; Whoosh or a hand-rolled inverted index in Python; pytest for the eval harness; everything pinned in a single `pyproject.toml`. Compute on a single workstation with one consumer GPU; no cloud required.

---

## 11. What I'm Asking the Instructor To Approve

Permission to file this under **Option 9** (with optional Option 10 LLM-query layer as a stretch). Confirmation that an off-the-shelf detector counts as a permitted "software library" under rubric requirement #7 (clearly attributed in the writeup). No team scaling beyond one person, so the April 20 multi-person justification doesn't apply.

---

*Next step after approval: scaffold the repo, freeze the held-out splits, and submit week-1 deliverable.*
