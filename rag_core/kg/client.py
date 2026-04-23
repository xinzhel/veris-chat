"""
Neo4j Knowledge Graph client for Victoria Unearthed parcel data.

Wraps the Neo4j Python driver to query parcel assessments and PDF report URLs.
Connection config is read from config.yaml under the 'neo4j' key.

Usage:
    from rag_core.kg.client import get_kg_client

    client = get_kg_client()
    urls = client.get_document_urls("433375739")
    context = client.get_parcel_context("433375739")
    client.close()
"""

import logging
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

from rag_core.chat.config import load_config

logger = logging.getLogger(__name__)

# Module-level cache
_cached_client: Optional["KGClient"] = None


def _unpack(value: Any) -> Any:
    """Unpack Neo4j array values (n10s stores all properties as arrays)."""
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


class KGClient:
    """
    Neo4j knowledge graph client for parcel data retrieval.

    All properties in the KG are stored as arrays due to n10s `handleMultival: ARRAY`.
    Use _unpack() to extract single values.
    """

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"[KG] Connected to Neo4j at {uri}")

    def close(self):
        """Close the Neo4j driver connection."""
        self._driver.close()
        logger.info("[KG] Neo4j connection closed")

    def get_document_urls(self, parcel_id: str) -> List[str]:
        """
        Get all assessment report PDF URLs for a parcel.

        Queries: Parcel -> hasOnsiteAssessment -> Resource
                 -> hasAssessmentReport -> AssessmentReport.hasLink

        Args:
            parcel_id: Parcel PFI identifier.

        Returns:
            List of PDF URL strings. Empty list if parcel not found or no reports.
        """
        query = """
            MATCH (p:Parcel)-[:hasOnsiteAssessment]->(a:Resource)
                  -[:hasAssessmentReport]->(r:AssessmentReport)
            WHERE $pfi IN p.hasPFI
            RETURN r.hasLink[0] AS pdf_url
        """
        with self._driver.session() as session:
            result = session.run(query, pfi=parcel_id)
            urls = [r["pdf_url"] for r in result if r["pdf_url"]]

        logger.info(f"[KG] get_document_urls({parcel_id}): {len(urls)} URLs")
        return urls

    def get_parcel_context(self, parcel_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all 7 assessment connection types for a parcel in a single query.

        Returns dict with keys: audits, licences, prsa, psr, vlr, overlays, business_listings.
        Each value is a list of dicts. Empty list means confirmed absence (queried but no data).

        Args:
            parcel_id: Parcel PFI identifier.

        Returns:
            Dict with 7 keys, each mapping to a list of assessment dicts.
        """
        # Single query for all assessment types (inspired by Oz's ALL_DETAILS_QUERY)
        query = """
            MATCH (p:Parcel)-[rel:hasOnsiteAssessment]->(a:Resource)
            WHERE $pfi IN p.hasPFI
            RETURN type(rel) AS rel_type,
                   [l IN labels(a) WHERE l <> 'Resource'][0] AS category,
                   properties(a) AS props
        """
        
        # Label → context key mapping
        label_map = {
            "EnvironmentalAudit": "audits",
            "EPALicence": "licences",
            "PreliminaryRiskScreeningAssessment": "prsa",
            "PrioritySiteRegister": "psr",
            "LandfillRegister": "vlr",
            "Overlay": "overlays",
            "HistoricalBusinessListing": "business_listings",
            "GroundwaterQualityRestrictedUseZone": "gqruz",
        }
        
        # Initialize all 7 keys with empty lists
        context: Dict[str, List[Dict[str, Any]]] = {v: [] for v in label_map.values()}
        
        with self._driver.session() as session:
            for record in session.run(query, pfi=parcel_id):
                category = record["category"]
                key = label_map.get(category)
                if key is None:
                    continue  # Skip unknown types (Geometry, etc.)
                
                props = record["props"]
                entry = {"relationship": record["rel_type"]}
                for k, v in props.items():
                    if k != "uri":
                        entry[k] = _unpack(v)
                context[key].append(entry)

        total = sum(len(v) for v in context.values())
        logger.info(f"[KG] get_parcel_context({parcel_id}): {total} assessments across 7 types (single query)")
        return context

    def _query_by_label(
        self, parcel_id: str, label: str
    ) -> List[Dict[str, Any]]:
        """
        Query assessments of a specific type for a parcel.

        Args:
            parcel_id: Parcel PFI identifier.
            label: Neo4j node label (e.g., "EnvironmentalAudit").

        Returns:
            List of dicts with assessment properties + relationship type.
        """
        query = f"""
            MATCH (p:Parcel)-[rel:hasOnsiteAssessment]->(a:{label})
            WHERE $pfi IN p.hasPFI
            RETURN type(rel) AS rel_type, properties(a) AS props
        """
        results = []
        with self._driver.session() as session:
            for record in session.run(query, pfi=parcel_id):
                props = record["props"]
                entry = {"relationship": record["rel_type"]}
                for k, v in props.items():
                    if k != "uri":
                        entry[k] = _unpack(v)
                results.append(entry)

        return results


def get_kg_client() -> KGClient:
    """
    Get or create a cached KGClient instance.

    Reads neo4j config from config.yaml. Caches the client for reuse.

    Returns:
        KGClient instance.
    """
    global _cached_client
    if _cached_client is None:
        config = load_config()
        neo4j_cfg = config.get("neo4j", {})
        _cached_client = KGClient(
            uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
            user=neo4j_cfg.get("user", "neo4j"),
            password=neo4j_cfg.get("password", "neo4jpassword"),
        )
    return _cached_client
