const express = require("express");
const router = express.Router();

/* ── Summary query: groups assessments by category with counts ── */
const SUMMARY_QUERY = `
  MATCH (p:Parcel)
  WHERE $pfi IN p.hasPFI
  MATCH (p)-[rel:hasOnsiteAssessment|hasOffsiteAssessment]->(a:Resource)
  WITH
    type(rel) AS relation,
    [l IN labels(a) WHERE l <> 'Resource'][0] AS category,
    a
  RETURN
    relation,
    category,
    count(a) AS count,
    collect(DISTINCT coalesce(a.hasBusinessType[0], a.hasPermissionType[0], a.hasRestrictedUse[0]))[0..3] AS sampleNames,
    [x IN collect(DISTINCT a.isHighPotentialContamination[0]) WHERE x = true]  <> [] AS hasHighRisk,
    [x IN collect(DISTINCT a.isMediumPotentialContamination[0]) WHERE x = true] <> [] AS hasMediumRisk
  ORDER BY relation, count DESC
`;

/* ── Detail query: individual reports for a specific category ── */
const DETAIL_QUERY = `
  MATCH (p:Parcel)
  WHERE $pfi IN p.hasPFI
  MATCH (p)-[rel:hasOnsiteAssessment|hasOffsiteAssessment]->(a:Resource)
  WHERE [l IN labels(a) WHERE l <> 'Resource'][0] = $category
    AND type(rel) = $relation
  OPTIONAL MATCH (a)-[:hasAssessmentReport]->(r:AssessmentReport)
  WITH
    rel, a,
    collect(CASE WHEN r IS NOT NULL AND r.hasLink IS NOT NULL THEN r.hasLink[0] ELSE null END) AS reportLinks
  RETURN
    type(rel) AS relation,
    [l IN labels(a) WHERE l <> 'Resource'][0] AS assessmentType,
    a.uri AS uri,
    a.assessmentDate AS assessmentDate,
    a.hasContaminationActivity AS hasContaminationActivity,
    a.hasBusinessType AS hasBusinessType,
    a.hasPermissionType AS hasPermissionType,
    a.hasRestrictedUse AS hasRestrictedUse,
    a.hasOverlayType AS hasOverlayType,
    a.isHighPotentialContamination AS isHighPotentialContamination,
    a.isMediumPotentialContamination AS isMediumPotentialContamination,
    coalesce(
      head([x IN reportLinks WHERE x IS NOT NULL]),
      a.hasLink[0]
    ) AS url
  ORDER BY a.assessmentDate DESC
`;

function unwrap(val) {
  if (val === null || val === undefined) return val;
  if (typeof val === "object" && val.low !== undefined && val.high !== undefined) {
    return val.toNumber ? val.toNumber() : val.low;
  }
  return val;
}

function unwrapRecord(obj) {
  const result = {};
  for (const [key, val] of Object.entries(obj)) {
    if (val && typeof val === "object" && val.properties) {
      result[key] = unwrapRecord(val.properties);
    } else if (Array.isArray(val)) {
      result[key] = val.map(unwrap);
    } else {
      result[key] = unwrap(val);
    }
  }
  return result;
}

function toStringArray(value) {
  if (Array.isArray(value)) return value.filter(Boolean).map((v) => String(v));
  if (value === null || value === undefined || value === "") return [];
  return [String(value)];
}

function firstString(value) {
  return toStringArray(value)[0] || undefined;
}

function toBoolArray(value) {
  if (!Array.isArray(value)) return [];
  return value.map((v) => v === true || v === 1 || v === "true");
}

function isRealUrl(value) {
  if (!value) return false;
  const s = String(value);
  return s.startsWith("http://") || s.startsWith("https://");
}

function toUrl(value) {
  return isRealUrl(value) ? String(value) : "";
}

function isAuthError(err) {
  const code = err?.code || "";
  const message = String(err?.message || "").toLowerCase();
  return (
    code.includes("Security.Unauthorized") ||
    message.includes("authentication failure") ||
    message.includes("incorrect authentication details") ||
    message.includes("unauthorized")
  );
}

/* ── GET /api/parcel-reports/:pfi — returns grouped summary ── */
router.get("/:pfi", async (req, res) => {
  const pfi = String(req.params.pfi || "").trim();
  if (!pfi) return res.status(400).json({ error: "PFI is required" });

  const session = req.app.locals.neo4jDriver.session();
  try {
    const result = await session.run(SUMMARY_QUERY, { pfi });

    const categories = result.records.map((record) => {
      const raw = unwrapRecord(record.toObject());
      return {
        relation: String(raw.relation || "related"),
        category: String(raw.category || "Resource"),
        count: Number(raw.count) || 0,
        sampleNames: toStringArray(raw.sampleNames).slice(0, 3),
        hasHighRisk: Boolean(raw.hasHighRisk),
        hasMediumRisk: Boolean(raw.hasMediumRisk),
      };
    });

    const totalCount = categories.reduce((sum, c) => sum + c.count, 0);
    res.json({ pfi, totalCount, categories });
  } catch (err) {
    if (isAuthError(err)) {
      return res.status(401).json({
        error: "Neo4j authentication failed. Check NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD.",
      });
    }
    console.error(`Error fetching summary for PFI ${pfi}:`, err);
    res.status(500).json({ error: err.message || "Failed to fetch parcel reports" });
  } finally {
    await session.close();
  }
});

/* ── GET /api/parcel-reports/:pfi/detail?category=X&relation=Y — expand a meta node ── */
router.get("/:pfi/detail", async (req, res) => {
  const pfi = String(req.params.pfi || "").trim();
  const category = String(req.query.category || "").trim();
  const relation = String(req.query.relation || "").trim();

  if (!pfi || !category || !relation) {
    return res.status(400).json({ error: "pfi, category, and relation are required" });
  }

  const session = req.app.locals.neo4jDriver.session();
  try {
    const result = await session.run(DETAIL_QUERY, { pfi, category, relation });

    const reports = result.records.map((record) => {
      const raw = unwrapRecord(record.toObject());
      const assessmentType = firstString(raw.assessmentType) || "Resource";
      return {
        relationshipType: firstString(raw.relation) || "related",
        assessmentType,
        url: toUrl(firstString(raw.url)),
        title:
          firstString(raw.hasBusinessType) ||
          firstString(raw.hasPermissionType) ||
          firstString(raw.hasRestrictedUse) ||
          firstString(raw.hasOverlayType) ||
          assessmentType,
        date: firstString(raw.assessmentDate),
        uri: firstString(raw.uri),
        hasContaminationActivity: toStringArray(raw.hasContaminationActivity),
        hasBusinessType: toStringArray(raw.hasBusinessType),
        isHighPotentialContamination: toBoolArray(raw.isHighPotentialContamination),
        isMediumPotentialContamination: toBoolArray(raw.isMediumPotentialContamination),
      };
    });

    res.json({ pfi, category, relation, reports });
  } catch (err) {
    if (isAuthError(err)) {
      return res.status(401).json({ error: "Neo4j authentication failed." });
    }
    console.error(`Error fetching detail for PFI ${pfi}/${category}:`, err);
    res.status(500).json({ error: err.message || "Failed to fetch detail" });
  } finally {
    await session.close();
  }
});

/* ── GET /api/parcel-reports/:pfi/all-details — all categories in one query ── */
const ALL_DETAILS_QUERY = `
  MATCH (p:Parcel)
  WHERE $pfi IN p.hasPFI
  MATCH (p)-[rel:hasOnsiteAssessment|hasOffsiteAssessment]->(a:Resource)
  OPTIONAL MATCH (a)-[:hasAssessmentReport]->(r:AssessmentReport)
  WITH
    type(rel) AS relation,
    [l IN labels(a) WHERE l <> 'Resource'][0] AS category,
    rel, a,
    collect(CASE WHEN r IS NOT NULL AND r.hasLink IS NOT NULL THEN r.hasLink[0] ELSE null END) AS reportLinks
  RETURN
    relation,
    category,
    type(rel) AS relationType,
    [l IN labels(a) WHERE l <> 'Resource'][0] AS assessmentType,
    a.uri AS uri,
    a.assessmentDate AS assessmentDate,
    a.hasContaminationActivity AS hasContaminationActivity,
    a.hasBusinessType AS hasBusinessType,
    a.hasPermissionType AS hasPermissionType,
    a.hasRestrictedUse AS hasRestrictedUse,
    a.hasOverlayType AS hasOverlayType,
    a.isHighPotentialContamination AS isHighPotentialContamination,
    a.isMediumPotentialContamination AS isMediumPotentialContamination,
    coalesce(
      head([x IN reportLinks WHERE x IS NOT NULL]),
      a.hasLink[0]
    ) AS url
  ORDER BY relation, category, a.assessmentDate DESC
`;

router.get("/:pfi/all-details", async (req, res) => {
  const pfi = String(req.params.pfi || "").trim();
  if (!pfi) return res.status(400).json({ error: "PFI is required" });

  const session = req.app.locals.neo4jDriver.session();
  try {
    const result = await session.run(ALL_DETAILS_QUERY, { pfi });

    // Group records by relation::category key
    const grouped = {};
    for (const record of result.records) {
      const raw = unwrapRecord(record.toObject());
      const key = `${raw.relation}::${raw.category}`;
      if (!grouped[key]) grouped[key] = [];
      const assessmentType = firstString(raw.assessmentType) || "Resource";
      grouped[key].push({
        relationshipType: firstString(raw.relationType) || "related",
        assessmentType,
        url: toUrl(firstString(raw.url)),
        title:
          firstString(raw.hasBusinessType) ||
          firstString(raw.hasPermissionType) ||
          firstString(raw.hasRestrictedUse) ||
          firstString(raw.hasOverlayType) ||
          assessmentType,
        date: firstString(raw.assessmentDate),
        uri: firstString(raw.uri),
        hasContaminationActivity: toStringArray(raw.hasContaminationActivity),
        hasBusinessType: toStringArray(raw.hasBusinessType),
        isHighPotentialContamination: toBoolArray(raw.isHighPotentialContamination),
        isMediumPotentialContamination: toBoolArray(raw.isMediumPotentialContamination),
      });
    }

    res.json({ pfi, details: grouped });
  } catch (err) {
    if (isAuthError(err)) {
      return res.status(401).json({ error: "Neo4j authentication failed." });
    }
    console.error(`Error fetching all-details for PFI ${pfi}:`, err);
    res.status(500).json({ error: err.message || "Failed to fetch all details" });
  } finally {
    await session.close();
  }
});

module.exports = router;
