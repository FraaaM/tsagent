"""Verdict protocol: how verdicts travel from the agent's text to ``anomaly_verdicts.jsonl``.

The agent has exactly two tools and no output tool (spec 02, FR3), so verdicts are carried
by its text: one JSON object per verdict, ``{"series_id": ..., "verdict": ...,
"explanation": ...}``. The parser here is deliberately lenient about *where* such objects
appear (bare lines, code fences, pretty-printed objects, arrays, surrounding prose) and
strict about *what* they must contain, which is exactly :class:`VerdictRecord`.

:class:`VerdictCollector` applies the episode rules on top of parsing: a verdict counts
only for a series the agent actually retrieved, the last verdict for a series wins, and
everything - accepted or rejected - is reported back so the trace can record it.
"""

from __future__ import annotations

import json
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Collection, Iterable

from pydantic import ValidationError

from .schemas import VERDICT_VALUES, VerdictRecord

_DECODER = json.JSONDecoder()
_VERDICT_KEYS = frozenset({"series_id", "verdict", "explanation"})
_ID_AFTER_KEY = re.compile(r"""series_id\W*["']([^"']+)["']""")


@dataclass(frozen=True)
class ParsedVerdict:
    """One verdict candidate found in the text."""

    raw: dict[str, Any]
    record: VerdictRecord | None
    error: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def series_id(self) -> str:
        sid = self.raw.get("series_id")
        return sid if isinstance(sid, str) else str(sid)


def extract_json_values(text: str) -> list[Any]:
    """Every JSON object or array embedded in ``text``, in order of appearance.

    Scans for ``{`` / ``[`` and attempts a decode from each; a successful decode consumes
    its span, so nested structures are returned once, at the outermost level.
    """
    values: list[Any] = []
    pos = 0
    n = len(text)
    while pos < n:
        if text[pos] not in "{[":
            pos += 1
            continue
        try:
            value, end = _DECODER.raw_decode(text, pos)
        except json.JSONDecodeError:
            pos += 1
            continue
        values.append(value)
        pos = end
    return values


def _candidates(values: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for v in values:
        if isinstance(v, dict):
            if "series_id" in v:
                out.append(v)
            else:
                out.extend(_candidates(v.values()))
        elif isinstance(v, list):
            out.extend(_candidates(v))
    return out


def _normalise(raw: dict[str, Any]) -> tuple[VerdictRecord | None, str | None, tuple[str, ...]]:
    notes: list[str] = []
    sid = raw.get("series_id")
    if not isinstance(sid, str) or not sid.strip():
        return None, "series_id must be a non-empty string", ()
    verdict = raw.get("verdict")
    if isinstance(verdict, str):
        cleaned = verdict.strip().lower()
        if cleaned != verdict:
            notes.append("verdict value normalised")
        verdict = cleaned
    if verdict not in VERDICT_VALUES:
        return None, f"verdict must be one of {list(VERDICT_VALUES)}, got {verdict!r}", tuple(notes)
    explanation = raw.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        explanation = json.dumps(explanation, ensure_ascii=False)
        notes.append("explanation coerced to text")
    if verdict == "normal" and explanation is not None:
        explanation = None
        notes.append("explanation dropped for a normal verdict")
    extra = sorted(set(raw) - _VERDICT_KEYS)
    if extra:
        notes.append(f"ignored extra field(s) {extra}")
    try:
        record = VerdictRecord(series_id=sid.strip(), verdict=verdict, explanation=explanation)
    except ValidationError as exc:
        return None, exc.errors()[0]["msg"], tuple(notes)
    return record, None, tuple(notes)


def parse_verdicts(text: str) -> list[ParsedVerdict]:
    """All verdict candidates in ``text``.

    Candidates are JSON objects carrying a ``series_id`` key (anywhere in the text, at any
    nesting depth), plus any line that mentions ``series_id`` without belonging to a
    decodable object - a broken attempt, reported so the trace can count it.
    """
    parsed: list[ParsedVerdict] = []
    for raw in _candidates(extract_json_values(text)):
        record, error, notes = _normalise(raw)
        parsed.append(ParsedVerdict(raw=raw, record=record, error=error, notes=notes))
    decoded_ids = [p.raw.get("series_id") for p in parsed]
    for line in text.splitlines():
        if "series_id" not in line:
            continue
        if any(isinstance(sid, str) and sid and sid in line for sid in decoded_ids):
            continue
        parsed.append(
            ParsedVerdict(
                raw={"series_id": _guess_id(line), "line": line.strip()},
                record=None,
                error="malformed verdict line (not valid JSON)",
            )
        )
    return parsed


def _guess_id(line: str) -> str:
    """Best-effort id for the trace: the first quoted token following ``series_id``."""
    match = _ID_AFTER_KEY.search(line)
    if match:
        return match.group(1).strip()
    tail = line.split("series_id", 1)[1]
    return tail.strip(" :=,\"'{}")[:80] or "?"


@dataclass
class IngestResult:
    accepted: list[VerdictRecord] = field(default_factory=list)
    rejected: list[tuple[ParsedVerdict, str]] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


class VerdictCollector:
    """Episode-level verdict state: which series were examined, which verdict stands."""

    def __init__(self) -> None:
        self.examined: OrderedDict[str, int] = OrderedDict()
        self.final: OrderedDict[str, VerdictRecord] = OrderedDict()
        self.revisions: Counter[str] = Counter()
        self.n_rejected = 0
        self.results: dict[int, IngestResult] = {}
        self._ingested_requests: set[int] = set()

    def mark_examined(self, series_id: str) -> None:
        self.examined[series_id] = self.examined.get(series_id, 0) + 1

    def has_ingested(self, request_index: int) -> bool:
        return request_index in self._ingested_requests

    def missing_verdicts(self) -> list[str]:
        """Examined series without a standing verdict, in examination order."""
        return [sid for sid in self.examined if sid not in self.final]

    def ingest(self, text: str, request_index: int, known_ids: Collection[str]) -> IngestResult:
        """Parse ``text`` and apply the episode rules. Idempotent per ``request_index``."""
        result = IngestResult()
        if request_index in self._ingested_requests:
            return result
        self._ingested_requests.add(request_index)
        self.results[request_index] = result
        if not text:
            return result
        for cand in parse_verdicts(text):
            sid = cand.series_id
            examined = sid in self.examined
            reason: str | None = None
            if cand.record is None:
                reason = f"invalid_record: {cand.error}"
            elif sid not in known_ids:
                reason = "unknown_series_id"
            elif not examined:
                reason = "not_examined"
            supersedes = False
            if reason is None:
                assert cand.record is not None
                supersedes = sid in self.final
                if supersedes:
                    self.revisions[sid] += 1
                    result.superseded.append(sid)
                self.final[sid] = cand.record
                result.accepted.append(cand.record)
            else:
                self.n_rejected += 1
                result.rejected.append((cand, reason))
            explanation = cand.raw.get("explanation")
            result.events.append(
                {
                    "request_index": request_index,
                    "series_id": sid,
                    "verdict": str(cand.raw.get("verdict")),
                    "explanation_chars": len(explanation) if isinstance(explanation, str) else 0,
                    "examined": examined,
                    "accepted": reason is None,
                    "reason": reason,
                    "supersedes": supersedes,
                }
            )
        return result

    def records(self) -> list[VerdictRecord]:
        return list(self.final.values())
