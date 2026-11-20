"""CSV and PDF reports of potential red-light violations.

Wording rules: an automated alert is always a "potential red-light
violation". Reviewer decisions are reported as reviewer decisions, not as
system accuracy. Footage that is not from Cape Coast is labelled as such.
"""
from __future__ import annotations

import csv
import io
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable, Optional

CSV_FIELDS = [
    "event_id", "analysis_id", "video", "location", "cape_coast_footage", "track_id", "vehicle_class",
    "outcome", "crossing_time_s", "crossing_frame", "signal_state_at_crossing", "red_onset_s",
    "red_onset_exact", "time_since_red_s", "detection_conf", "review_status", "reviewer",
    "review_note", "reviewed_at",
]

OUTCOME_LABELS = {
    "potential_violation": "Potential red-light violation",
    "signal_uncertain": "Crossing while signal unreadable",
    "lawful": "Crossing (not flagged)",
}


def events_csv(events: Iterable[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    w.writeheader()
    for e in events:
        w.writerow({
            "event_id": e["id"], "analysis_id": e["analysis_id"], "video": e.get("video_filename", ""),
            "location": e.get("video_location", ""),
            "cape_coast_footage": "yes" if e.get("video_is_cape_coast") else "no",
            "track_id": e["track_id"], "vehicle_class": e["vehicle_class"], "outcome": e["outcome"],
            "crossing_time_s": e["crossing_time_s"], "crossing_frame": e["crossing_frame"],
            "signal_state_at_crossing": e["signal_state"], "red_onset_s": e["red_onset_s"],
            "red_onset_exact": e["red_onset_exact"], "time_since_red_s": e["time_since_red_s"],
            "detection_conf": e["det_conf"], "review_status": e["review_status"], "reviewer": e["reviewer"],
            "review_note": e["review_note"], "reviewed_at": e["reviewed_at"],
        })
    return buf.getvalue()


def summarise(events: list[dict], segments: Optional[list[dict]] = None) -> dict:
    flagged = [e for e in events if e["outcome"] == "potential_violation"]
    reviewed = [e for e in flagged if e["review_status"] in ("confirmed", "rejected")]
    confirmed = sum(e["review_status"] == "confirmed" for e in flagged)
    out = {
        "crossings": len(events),
        "by_outcome": dict(Counter(e["outcome"] for e in events)),
        "potential_violations": len(flagged),
        "potential_violations_by_class": dict(Counter(e["vehicle_class"] for e in flagged)),
        "review_status": dict(Counter(e["review_status"] for e in flagged)),
        # Share of reviewed alerts that reviewers confirmed. This is NOT the
        # system's precision against annotated ground truth.
        "reviewer_confirmed_share": round(confirmed / len(reviewed), 4) if reviewed else None,
        "time_since_red_s": None,
    }
    tsr = sorted(e["time_since_red_s"] for e in flagged if e["time_since_red_s"] is not None)
    if tsr:
        out["time_since_red_s"] = {"min": tsr[0], "median": tsr[len(tsr) // 2], "max": tsr[-1]}
    if segments:
        dur = Counter()
        for s in segments:
            dur[s["state"]] += max(0.0, s["end_s"] - s["start_s"])
        total = sum(dur.values()) or 1.0
        out["signal_time_share"] = {k: round(v / total, 4) for k, v in dur.items()}
    return out


def events_pdf(analysis: dict, events: list[dict], summary: dict,
               snapshots: Optional[dict[int, str]] = None, anonymised: bool = False) -> bytes:
    """Render a PDF report. ``snapshots`` maps event id -> image path."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="Potential red-light violations - review report")
    ss = getSampleStyleSheet()
    small = ss["BodyText"].clone("small", fontSize=8, leading=10)
    story = [Paragraph("Potential red-light violations: review report", ss["Title"])]
    cape = analysis.get("video_is_cape_coast")
    source_note = ("Footage recorded in Cape Coast." if cape else
                   "<b>This footage is NOT marked as Cape Coast footage. Do not report these results as "
                   "Cape Coast results.</b>")
    story += [
        Paragraph(
            "Automated alerts in this report are <b>potential</b> red-light violations produced by a research "
            "prototype. An alert is not a violation until a person has reviewed the evidence. This report "
            "contains no number-plate recognition or driver identification and supports no enforcement "
            "decision.", ss["BodyText"]),
        Spacer(1, 6),
        Paragraph(source_note, ss["BodyText"]),
        Spacer(1, 10),
    ]
    meta = [
        ["Analysis", str(analysis["id"])],
        ["Video", analysis.get("video_filename", "")],
        ["Location", analysis.get("video_location", "") or "-"],
        ["Detector / tracker", analysis.get("model", "")],
        ["Analysis finished", analysis.get("finished_at") or "-"],
        ["Report generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")],
        ["Images anonymised", "yes" if anonymised else ("no images" if not snapshots else "no")],
    ]
    t = Table(meta, colWidths=[4.5 * cm, 12 * cm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke)]))
    story += [t, Spacer(1, 10), Paragraph("Summary", ss["Heading2"])]
    rs = summary.get("review_status", {})
    rows = [
        ["Stop-line crossings detected", summary["crossings"]],
        ["Potential red-light violations", summary["potential_violations"]],
        ["Crossings while signal unreadable", summary["by_outcome"].get("signal_uncertain", 0)],
        ["Reviewer decisions (confirmed / rejected / unsure / pending)",
         f"{rs.get('confirmed', 0)} / {rs.get('rejected', 0)} / {rs.get('unsure', 0)} / {rs.get('pending', 0)}"],
    ]
    if summary.get("reviewer_confirmed_share") is not None:
        rows.append(["Share of reviewed alerts confirmed by reviewers*",
                     f"{100 * summary['reviewer_confirmed_share']:.1f}%"])
    t = Table(rows, colWidths=[11 * cm, 5.5 * cm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    story += [t]
    if summary.get("reviewer_confirmed_share") is not None:
        story.append(Paragraph("* Reviewer decisions on this video only. This is not a measure of system "
                               "accuracy against annotated ground truth.", small))
    story += [Spacer(1, 10), Paragraph("Events", ss["Heading2"])]
    header = ["ID", "Time (s)", "Vehicle", "Outcome", "Signal", "After red (s)", "Review"]
    data = [header]
    for e in events:
        data.append([e["id"], f"{e['crossing_time_s']:.2f}", e["vehicle_class"],
                     Paragraph(OUTCOME_LABELS.get(e["outcome"], e["outcome"]), small), e["signal_state"],
                     "-" if e["time_since_red_s"] is None else f"{e['time_since_red_s']:.2f}",
                     e["review_status"]])
    t = Table(data, repeatRows=1, colWidths=[1.2 * cm, 1.8 * cm, 2.2 * cm, 4.6 * cm, 2 * cm, 2.2 * cm, 2.4 * cm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t)
    if snapshots:
        story.append(PageBreak())
        story.append(Paragraph("Evidence snapshots" + (" (anonymised)" if anonymised else ""), ss["Heading2"]))
        if not anonymised:
            story.append(Paragraph("These images are not anonymised. Do not share or publish them.", small))
        for e in events:
            p = snapshots.get(e["id"])
            if not p:
                continue
            story.append(Paragraph(
                f"Event {e['id']}: {OUTCOME_LABELS.get(e['outcome'], e['outcome'])}, track {e['track_id']}, "
                f"{e['vehicle_class']}, t = {e['crossing_time_s']:.2f} s, review: {e['review_status']}", small))
            story.append(Image(p, width=16 * cm, height=9 * cm, kind="proportional"))
            story.append(Spacer(1, 8))
    doc.build(story)
    return buf.getvalue()
