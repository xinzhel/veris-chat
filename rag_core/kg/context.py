"""
Parcel context formatting and session ID parsing.

Converts KGClient output into natural-language context for LLM system messages,
and parses session IDs in the format `parcel_id::temp_id`.

Usage:
    from rag_core.kg.context import format_parcel_context, parse_session_id

    parcel_id, temp_id = parse_session_id("433375739::abc123")
    context = kg_client.get_parcel_context(parcel_id)
    system_block = format_parcel_context(parcel_id, context)
"""

from typing import Any, Dict, List, Tuple


def parse_session_id(session_id: str) -> Tuple[str, str]:
    """
    Parse 'parcel_id::temp_id' into (parcel_id, temp_id).

    Args:
        session_id: Session ID in format 'parcel_id::temp_id'.

    Returns:
        Tuple of (parcel_id, temp_id).

    Raises:
        ValueError: If '::' separator is missing.
    """
    if "::" not in session_id:
        raise ValueError(
            f"Invalid session ID format: '{session_id}'. Expected 'parcel_id::temp_id'"
        )
    parcel_id, temp_id = session_id.split("::", 1)
    return parcel_id.strip(), temp_id.strip()


# Display names and formatters for each assessment type
_TYPE_CONFIG = {
    "audits": {
        "title": "Environmental Audits",
        "format": lambda item: _format_dated(item, extra_key="assessmentDate"),
    },
    "licences": {
        "title": "EPA Licences",
        "format": lambda item: _format_licence(item),
    },
    "prsa": {
        "title": "Preliminary Risk Screening Assessments (PRSA)",
        "format": lambda item: _format_dated(item, extra_key="assessmentDate"),
    },
    "psr": {
        "title": "Priority Site Register",
        "format": lambda item: _format_psr(item),
    },
    "vlr": {
        "title": "Landfill Register (VLR, 500m buffer)",
        "format": lambda item: _format_vlr(item),
    },
    "overlays": {
        "title": "Planning Overlays",
        "format": lambda item: _format_overlay(item),
    },
    "business_listings": {
        "title": "Historical Business Listings",
        "format": lambda item: _format_business(item),
    },
    "gqruz": {
        "title": "Groundwater Quality Restricted Use Zones (GQRUZ)",
        "format": lambda item: _format_gqruz(item),
    },
}


def format_parcel_context(
    parcel_id: str, kg_context: Dict[str, List[Dict[str, Any]]]
) -> str:
    """
    Format KG context dict into a natural-language system message block.

    Always includes all 7 assessment types. Empty types show "No data found"
    (confirmed absence, not missing info).

    Args:
        parcel_id: The parcel PFI.
        kg_context: Output of KGClient.get_parcel_context().

    Returns:
        Formatted string block for the LLM system message.
    """
    lines = [f"## Parcel Context (PFI: {parcel_id})", ""]

    for key, config in _TYPE_CONFIG.items():
        items = kg_context.get(key, [])
        lines.append(f"### {config['title']}")

        if not items:
            lines.append("- No data found for this parcel")
        else:
            for item in items:
                rel = item.get("relationship", "")
                prefix = "(offsite)" if rel == "hasOffsiteAssessment" else ""
                text = config["format"](item)
                if prefix:
                    text = f"{prefix} {text}"
                pdf_url = item.get("pdf_url")
                if pdf_url:
                    text = f"{text} | [PDF]({pdf_url})"
                lines.append(f"- {text}")

        lines.append("")

    return "\n".join(lines)


# --- Formatters for each assessment type ---


def _format_dated(item: Dict[str, Any], extra_key: str = "assessmentDate") -> str:
    date = item.get(extra_key, "unknown date")
    if date and date != "NaT":
        return f"Date: {date}"
    return "Date: unknown"


def _format_licence(item: Dict[str, Any]) -> str:
    ptype = item.get("hasPermissionType", "unknown type")
    date = item.get("assessmentDate", "")
    parts = [f"Type: {ptype}"]
    if date and date != "NaT":
        parts.append(f"Date: {date}")
    return ", ".join(parts)


def _format_psr(item: Dict[str, Any]) -> str:
    issue = item.get("hasIssue", "No details available")
    return f"Issue: {issue}"


def _format_vlr(item: Dict[str, Any]) -> str:
    waste = item.get("hasWasteType", "unknown waste type")
    return f"Waste type: {waste}"


def _format_overlay(item: Dict[str, Any]) -> str:
    otype = item.get("hasOverlayType", "unknown overlay")
    date = item.get("assessmentDate", "")
    parts = [otype]
    if date and date != "NaT":
        parts.append(f"({date})")
    return " ".join(parts)


def _format_business(item: Dict[str, Any]) -> str:
    btype = item.get("hasBusinessType", "unknown business")
    activity = item.get("hasContaminationActivity", "")
    high_risk = item.get("isHighPotentialContamination", False)
    date = item.get("assessmentDate", "")

    parts = [btype]
    if activity:
        parts.append(f"— {activity}")
    if high_risk:
        parts.append("(HIGH contamination risk)")
    if date and date != "NaT":
        parts.append(f"[{date}]")
    return " ".join(parts)


def _format_gqruz(item: Dict[str, Any]) -> str:
    restricted_use = item.get("hasRestrictedUse", "unknown")
    date = item.get("assessmentDate", "")
    parts = [f"Restricted use: {restricted_use}"]
    if date and date != "NaT":
        parts.append(f"({date})")
    return " ".join(parts)
