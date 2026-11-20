# Paper vs. software check (20 Nov 2026)

Paper checked: *Red_Light_Violation_Detection_Paper_1.docx*. Software: `/home/user/redlight` at commit 43c8af5.
Each item was verified in the code, not from memory.

## A. The publication plan

| Item | Status |
|---|---|
| Methods section in academic prose, no code details | **OK** (Sections 3–4) |
| Pipeline figure | **OK** (Figure 1 matches the system) |
| Appendix: table of final parameters | **Missing** |
| Code Availability statement (repository + Zenodo DOI) | **Missing** (Declarations has ethics, data, conflict of interest only) |
| Corresponding author e-mail | **Placeholder** still in the paper ("[insert email address]") |

## B. Things the paper says the system does that the code does differently or not at all

| # | Paper says (section) | Code now | Fix |
|---|---|---|---|
| 1 | Reference point is the **centre of the leading edge** of the box, the edge facing the junction (3.5, Table 3, 6.3) | Uses the **bottom-centre** of the box | Code: direction-aware leading-edge point |
| 2 | Vehicle must go a **confirmation distance** past the line; a vehicle that edges over and stops is not counted (3.5 rule 2) | Only a 3-pixel jitter band; edging over by more than 3 px on red is flagged | Code: configurable confirmation distance |
| 3 | A crossing **within a small tolerance of red onset** is saved as **uncertain** (3.5) | Treated as lawful (inside `red_grace_s`) or as a potential violation (grace 0) | Code: onset tolerance → uncertain |
| 4 | Tracks lost near the line, or re-appearing under a new ID, are **counted and reported** (3.5) | Not counted | Code: count and report them |
| 5 | Clip of **about two seconds** either side (3.6) | Default 3 s | Code default 2 s (or edit paper) |
| 6 | Event has a **certainty indicator** for the signal state (3.6) | Signal state only | Code: store lamp scores / onset exactness with each event |
| 7 | Review statuses **unreviewed / confirmed / rejected / uncertain** (3.6) | pending / confirmed / rejected / unsure | Code: rename labels |
| 8 | Dashboard shows the **processed video** with boxes, IDs, signal and event markers, and a **count of unique tracked vehicles by class** (3.8) | Only event clips; no vehicle counts | Code: add both |
| 9 | Each run records the **hardware** (3.8) | Not recorded per run | Code: add to run stats |
| 10 | **Six classes**: car/taxi, minibus (trotro), bus, truck, motorcycle, tricycle (3.4) | Dataset script hard-codes 4 COCO classes; privacy rider band for motorcycles only | Code: six-class dataset config; tricycle handled |
| 11 | Event match needs **same approach and same vehicle** (4.3) | Time only | Code: optional class/approach match |
| 12 | Scores reported **twice**: uncertain set aside, and uncertain counted as positive (4.3) | Only the first | Code: add second variant |
| 13 | **RLR rate per vehicle and per red phase**, automated vs manual (4.3) | No script | Code: add script |
| 14 | Tracking: **ID switches per 100 vehicles; share tracked through the stop-line zone** (Table 5) | MOTA, IDF1, raw ID switches | Code: add both measures |
| 15 | Signal: **delay in detecting the start of red** (Table 5) | Not computed | Code: add |
| 16 | **Bootstrap 95% CIs** over test clips and **McNemar's exact test** (4.6, Table 3, 5.1 says "written") | Neither exists (Wilson intervals instead) | Code: add both |
| 17 | Computational table: **parameters, GFLOPs, model size, peak memory** (Table 6) | Only ms/frame and FPS | Code: add |
| 18 | **Condition tags**: density from vehicles/frame (thirds), lighting from time + brightness; metrics by tag (4.5) | No script | Code: add tagging + breakdown |
| 19 | **Leave-one-site-out** experiment (4.1) | Possible by hand only | Code: option in split script |
| 20 | Section 5.1: "**42 tests**"; synthetic video with a crossing **0.05 s after red**, a vehicle **first seen inside the junction**, a vehicle that **edged over on red and went at green**; result **2 potential, 2 uncertain, 4 lawful, 2 ignored** | 113 tests; the synthetic video has different cases (3 potential, 1 uncertain, 7 lawful, 1 ignored) | Code: add the paper's cases after 1–3; then update 5.1 numbers from the real run |
| 21 | Accident demo: "the system would … **mark the approximate time** of the collision" (3.9) | A **person** marks it | Paper: say a person marks it |

## C. Built but not in the paper (decide whether to mention)

Signal calibration from labelled frames and the peak-brightness safety guard (used if you calibrate: should be in 3.5); anonymisation details and the human sign-off (fits 3.10); BoT-SORT option, OpenVINO export, CNN signal classifier (mention only if used for the reported results); Ultralytics analytics switched off (fits 3.10).

## D. Not affected

Literature review, references (every in-text citation has an entry), research questions, ethics, the decision rule's core logic, the temporal signal filter, session-level split, three seeds, YOLO11n vs YOLO11m, ByteTrack.
