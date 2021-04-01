"""
Layer 1 — Security Data Fabric: Schema Normalization
Transforms vendor-specific event formats into OCSF canonical events.
Complexity: O(n) time, O(n) space. Target: 10M+ events/day.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from src.common.models import EventSource, OCSFEvent

logger = logging.getLogger(__name__)

# Vendor field mapping registry: source → {vendor_field: ocsf_field}
_FIELD_MAPS: Dict[EventSource, Dict[str, str]] = {
    EventSource.CROWDSTRIKE: {
        "ComputerName": "actor_device",
        "UserName": "actor_user",
        "LocalIP": "src_ip",
        "RemoteIP": "dst_ip",
        "CommandLine": "activity_name",
        "Severity": "severity",
    },
    EventSource.OKTA: {
        "actor.alternateId": "actor_user",
        "client.ipAddress": "src_ip",
        "eventType": "activity_name",
        "target[0].alternateId": "target_resource",
        "severity": "severity",
    },
    EventSource.AWS_CLOUDTRAIL: {
        "userIdentity.userName": "actor_user",
        "sourceIPAddress": "src_ip",
        "eventName": "activity_name",
        "requestParameters.resourceId": "target_resource",
    },
    EventSource.CISCO: {
        "src_ip": "src_ip",
        "dst_ip": "dst_ip",
        "user": "actor_user",
        "event_type": "activity_name",
        "severity": "severity",
    },
    EventSource.TENABLE: {
        "plugin.name": "activity_name",
        "host.hostname": "actor_device",
        "plugin.cve": "cve_ids",
        "severity": "severity",
        "plugin.solution": "metadata",
    },
}

_SEVERITY_MAP: Dict[str, int] = {
    "info": 1, "informational": 1, "low": 2, "medium": 3,
    "high": 4, "critical": 5, "unknown": 1,
}

_CATEGORY_MAP: Dict[EventSource, str] = {
    EventSource.CROWDSTRIKE: "endpoint",
    EventSource.OKTA: "identity",
    EventSource.AWS_CLOUDTRAIL: "cloud",
    EventSource.CISCO: "network",
    EventSource.TENABLE: "vulnerability",
}


def _resolve_nested(data: Dict[str, Any], key: str) -> Any:
    """Resolve dot-notation keys like 'actor.alternateId' and array index keys."""
    parts = key.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict):
            # handle array notation: target[0]
            if "[" in part:
                name, idx = part.rstrip("]").split("[")
                current = current.get(name, [])
                try:
                    current = current[int(idx)]
                except (IndexError, TypeError):
                    return None
            else:
                current = current.get(part)
        else:
            return None
    return current


def normalize(raw: Dict[str, Any], source: EventSource) -> OCSFEvent:
    """Map a single vendor event dict to an OCSFEvent."""
    field_map = _FIELD_MAPS.get(source, {})
    event = OCSFEvent(
        source=source,
        category=_CATEGORY_MAP.get(source, "generic"),
        raw_payload=raw,
    )

    # parse timestamp
    for ts_key in ("timestamp", "EventTime", "eventTime", "occurred", "CreatedAt"):
        ts_val = _resolve_nested(raw, ts_key)
        if ts_val:
            try:
                if isinstance(ts_val, (int, float)):
                    event.timestamp = datetime.utcfromtimestamp(ts_val / 1000 if ts_val > 1e10 else ts_val)
                else:
                    event.timestamp = datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
                break
            except (ValueError, OSError):
                pass

    for vendor_field, ocsf_field in field_map.items():
        val = _resolve_nested(raw, vendor_field)
        if val is None:
            continue
        if ocsf_field == "severity":
            if isinstance(val, str):
                event.severity = _SEVERITY_MAP.get(val.lower(), 1)
            elif isinstance(val, int):
                event.severity = max(1, min(5, val))
        elif ocsf_field == "cve_ids":
            event.cve_ids = val if isinstance(val, list) else [val]
        elif ocsf_field == "metadata":
            event.metadata[vendor_field] = val
        else:
            setattr(event, ocsf_field, str(val))

    return event


class SDFNormalizer:
    """
    Streaming normalizer. Processes raw events from a Kafka consumer,
    maps them to OCSF, and publishes to the normalized topic.
    Runs as an async Kafka consumer group member.
    """

    def __init__(self, custom_maps: Optional[Dict[EventSource, Dict[str, str]]] = None) -> None:
        if custom_maps:
            for src, mapping in custom_maps.items():
                _FIELD_MAPS.setdefault(src, {}).update(mapping)
        self._processed = 0
        self._errors = 0

    def process(self, raw_json: str, source_hint: Optional[str] = None) -> Optional[OCSFEvent]:
        try:
            raw = json.loads(raw_json)
            source = EventSource(source_hint) if source_hint else self._detect_source(raw)
            event = normalize(raw, source)
            self._processed += 1
            return event
        except Exception as exc:
            self._errors += 1
            logger.warning("Normalization failed for source=%s: %s", source_hint, exc)
            return None

    @staticmethod
    def _detect_source(raw: Dict[str, Any]) -> EventSource:
        """Heuristic source detection when no explicit hint provided."""
        keys = set(raw.keys())
        if "ComputerName" in keys or "FalconHostLink" in keys:
            return EventSource.CROWDSTRIKE
        if "actor" in keys and "eventType" in keys:
            return EventSource.OKTA
        if "eventSource" in raw and "amazonaws" in str(raw["eventSource"]):
            return EventSource.AWS_CLOUDTRAIL
        if "plugin" in keys and "host" in keys:
            return EventSource.TENABLE
        return EventSource.GENERIC

    @property
    def stats(self) -> Dict[str, int]:
        return {"processed": self._processed, "errors": self._errors}
