```bash
(.venv) (.venv) veris-chat $ source .venv/bin/activate && python -c "
cmdand dquote> from rag_core.kg.client import get_kg_client
cmdand dquote> client = get_kg_client()
cmdand dquote> query = '''
cmdand dquote>     MATCH (p:Parcel)-[rel:hasOnsiteAssessment]->(a:Resource)
cmdand dquote>     WHERE \$pfi IN p.hasPFI
cmdand dquote>     RETURN type(rel) AS rel_type,
cmdand dquote>            [l IN labels(a) WHERE l <> \"Resource\"][0] AS category,
cmdand dquote>            properties(a) AS props
cmdand dquote> '''
cmdand dquote> with client._driver.session() as session:
cmdand dquote>     for r in session.run(query, pfi='210470171'):
cmdand dquote>         props = r['props']
cmdand dquote>         # Check if any prop value looks like a URL
cmdand dquote>         url_keys = [k for k, v in props.items() if isinstance(v, (str, list)) and 'http' in str(v)]
cmdand dquote>         print(f\"{r['category']}: url_keys={url_keys}, all_keys={list(props.keys())}\")
cmdand dquote> client.close()
cmdand dquote> "
GroundwaterQualityRestrictedUseZone: url_keys=['uri'], all_keys=['hasRestrictedUse', 'assessmentDate', 'uri']
GroundwaterQualityRestrictedUseZone: url_keys=['uri'], all_keys=['hasRestrictedUse', 'assessmentDate', 'uri']
Overlay: url_keys=['uri'], all_keys=['hasOverlayType', 'assessmentDate', 'uri']
Overlay: url_keys=['uri'], all_keys=['hasOverlayType', 'assessmentDate', 'uri']
GroundwaterQualityRestrictedUseZone: url_keys=['uri'], all_keys=['hasRestrictedUse', 'assessmentDate', 'uri']
Overlay: url_keys=['uri'], all_keys=['hasOverlayType', 'assessmentDate', 'uri']
Overlay: url_keys=['uri'], all_keys=['hasOverlayType', 'assessmentDate', 'uri']
EnvironmentalAudit: url_keys=['uri'], all_keys=['assessmentDate', 'uri']
Overlay: url_keys=['uri'], all_keys=['hasOverlayType', 'assessmentDate', 'uri']
Overlay: url_keys=['uri'], all_keys=['hasOverlayType', 'assessmentDate', 'uri']
HistoricalBusinessListing: url_keys=['uri'], all_keys=['assessmentDate', 'isHighPotentialContamination', 'hasContaminationActivity', 'isMediumPotentialContamination', 'uri', 'hasBusinessType']
Overlay: url_keys=['uri'], all_keys=['hasOverlayType', 'assessmentDate', 'uri']
GroundwaterQualityRestrictedUseZone: url_keys=['uri'], all_keys=['hasRestrictedUse', 'assessmentDate', 'uri']
GroundwaterQualityRestrictedUseZone: url_keys=['uri'], all_keys=['hasRestrictedUse', 'assessmentDate', 'uri']
Overlay: url_keys=['uri'], all_keys=['hasOverlayType', 'assessmentDate', 'uri']
Overlay: url_keys=['uri'], all_keys=['hasOverlayType', 'assessmentDate', 'uri']
HistoricalBusinessListing: url_keys=['uri'], all_keys=['assessmentDate', 'isHighPotentialContamination', 'hasContaminationActivity', 'isMediumPotentialContamination', 'uri', 'hasBusinessType']
GroundwaterQualityRestrictedUseZone: url_keys=['uri'], all_keys=['hasRestrictedUse', 'assessmentDate', 'uri']
(.venv) (.venv) veris-chat $ source .venv/bin/activate && python -c "
cmdand dquote> from rag_core.kg.client import get_kg_client
cmdand dquote> from rag_core.kg.context import format_parcel_context
cmdand dquote> 
cmdand dquote> client = get_kg_client()
cmdand dquote> context = client.get_parcel_context('210470171')
cmdand dquote> print(format_parcel_context('210470171', context))
cmdand dquote> client.close()
cmdand dquote> "
## Parcel Context (PFI: 210470171)

### Environmental Audits
- Date: 2008-07-03 00:00:00

### EPA Licences
- No data found for this parcel

### Preliminary Risk Screening Assessments (PRSA)
- No data found for this parcel

### Priority Site Register
- No data found for this parcel

### Landfill Register (VLR, 500m buffer)
- No data found for this parcel

### Planning Overlays
- DESIGN AND DEVELOPMENT OVERLAY - SCHEDULE 12 (2017-06-07 00:00:00)
- PARKING OVERLAY - PRECINCT 7 (2017-06-07 00:00:00)
- DEVELOPMENT PLAN OVERLAY - SCHEDULE 4 (2017-06-07 00:00:00)
- PARKING OVERLAY - PRECINCT 7 (2017-06-07 00:00:00)
- DESIGN AND DEVELOPMENT OVERLAY - SCHEDULE 51 (2017-06-07 00:00:00)
- DESIGN AND DEVELOPMENT OVERLAY - SCHEDULE 51 (AREA 4) (2017-06-07 00:00:00)
- DESIGN AND DEVELOPMENT OVERLAY - SCHEDULE 1 (2021-09-27 00:00:00)
- PARKING OVERLAY - PRECINCT 1 (2018-09-18 00:00:00)
- DESIGN AND DEVELOPMENT OVERLAY - SCHEDULE 10 (2020-12-04 00:00:00)

### Historical Business Listings
- Agents - Tourists — Unknown [1896-01-01 00:00:00]
- Service Stations — Service stations/fuel storage (HIGH contamination risk) [1965-01-01 00:00:00]

### Groundwater Quality Restricted Use Zones (GQRUZ)
- Restricted use: Drinking water;Irrigation of crops (including domestic gardens) and parks;Livestock water supply;Water used for recreational purposes (e.g. swimming)
- Restricted use: Drinking water;Livestock water supply;Water used for recreational purposes (e.g. swimming)
- Restricted use: Drinking water;Water used for recreational purposes (e.g. swimming)
- Restricted use: Drinking water;Water used for recreational purposes (e.g. swimming)
- Restricted use: Drinking water;Livestock water supply;Water used for recreational purposes (e.g. swimming)
- Restricted use: Drinking water;Irrigation of crops (including domestic gardens) and parks;Livestock water supply;Water used for recreational purposes (e.g. swimming)

(.venv) (.venv) veris-chat $ source .venv/bin/activate && python -c "
cmdand dquote> from rag_core.kg.client import get_kg_client
cmdand dquote> import json
cmdand dquote> 
cmdand dquote> client = get_kg_client()
cmdand dquote> query = '''
cmdand dquote>     MATCH (p:Parcel)-[rel:hasOnsiteAssessment]->(a:Resource)
cmdand dquote>     WHERE \$pfi IN p.hasPFI
cmdand dquote>     RETURN type(rel) AS rel_type,
cmdand dquote>            [l IN labels(a) WHERE l <> \"Resource\"][0] AS category,
cmdand dquote>            properties(a) AS props
cmdand dquote> '''
cmdand dquote> with client._driver.session() as session:
cmdand dquote>     for r in session.run(query, pfi='210470171'):
cmdand dquote>         if r['category'] == 'GroundwaterQualityRestrictedUseZone':
cmdand dquote>             print(json.dumps({
cmdand dquote>                 'rel_type': r['rel_type'],
cmdand dquote>                 'category': r['category'],
cmdand dquote>                 'props': {k: (list(v) if isinstance(v, list) else v) for k, v in r['props'].items()}
cmdand dquote>             }, indent=2, default=str))
cmdand dquote>             print('---')
cmdand dquote> client.close()
cmdand dquote> "
{
  "rel_type": "hasOnsiteAssessment",
  "category": "GroundwaterQualityRestrictedUseZone",
  "props": {
    "hasRestrictedUse": [
      "Drinking water;Irrigation of crops (including domestic gardens) and parks;Livestock water supply;Water used for recreational purposes (e.g. swimming)"
    ],
    "assessmentDate": [
      "NaT"
    ],
    "uri": "http://example.org/data/vic-unearthed/groundwater/7000096"
  }
}
---
{
  "rel_type": "hasOnsiteAssessment",
  "category": "GroundwaterQualityRestrictedUseZone",
  "props": {
    "hasRestrictedUse": [
      "Drinking water;Livestock water supply;Water used for recreational purposes (e.g. swimming)"
    ],
    "assessmentDate": [
      "NaT"
    ],
    "uri": "http://example.org/data/vic-unearthed/groundwater/7000158"
  }
}
---
{
  "rel_type": "hasOnsiteAssessment",
  "category": "GroundwaterQualityRestrictedUseZone",
  "props": {
    "hasRestrictedUse": [
      "Drinking water;Water used for recreational purposes (e.g. swimming)"
    ],
    "assessmentDate": [
      "NaT"
    ],
    "uri": "http://example.org/data/vic-unearthed/groundwater/7000248"
  }
}
---
{
  "rel_type": "hasOnsiteAssessment",
  "category": "GroundwaterQualityRestrictedUseZone",
  "props": {
    "hasRestrictedUse": [
      "Drinking water;Water used for recreational purposes (e.g. swimming)"
    ],
    "assessmentDate": [
      "NaT"
    ],
    "uri": "http://example.org/data/vic-unearthed/groundwater/7000175"
  }
}
---
{
  "rel_type": "hasOnsiteAssessment",
  "category": "GroundwaterQualityRestrictedUseZone",
  "props": {
    "hasRestrictedUse": [
      "Drinking water;Livestock water supply;Water used for recreational purposes (e.g. swimming)"
    ],
    "assessmentDate": [
      "NaT"
    ],
    "uri": "http://example.org/data/vic-unearthed/groundwater/7000118"
  }
}
---
{
  "rel_type": "hasOnsiteAssessment",
  "category": "GroundwaterQualityRestrictedUseZone",
  "props": {
    "hasRestrictedUse": [
      "Drinking water;Irrigation of crops (including domestic gardens) and parks;Livestock water supply;Water used for recreational purposes (e.g. swimming)"
    ],
    "assessmentDate": [
      "NaT"
    ],
    "uri": "http://example.org/data/vic-unearthed/groundwater/7000073"
  }
}
---
(.venv) (.venv) veris-chat $ source .venv/bin/activate && python -c "
cmdand dquote> from rag_core.kg.client import get_kg_client
cmdand dquote> import json
cmdand dquote> 
cmdand dquote> client = get_kg_client()
cmdand dquote> query = '''
cmdand dquote>     MATCH (p:Parcel)-[rel:hasOnsiteAssessment]->(a:Resource)
cmdand dquote>           -[:hasAssessmentReport]->(r:AssessmentReport)
cmdand dquote>     WHERE \$pfi IN p.hasPFI
cmdand dquote>     AND [l IN labels(a) WHERE l <> \"Resource\"][0] = \"GroundwaterQualityRestrictedUseZone\"
cmdand dquote>     RETURN [l IN labels(a) WHERE l <> \"Resource\"][0] AS category,
cmdand dquote>            properties(r) AS report_props
cmdand dquote> '''
cmdand dquote> with client._driver.session() as session:
cmdand dquote>     for rec in session.run(query, pfi='210470171'):
cmdand dquote>         print(json.dumps({
cmdand dquote>             'category': rec['category'],
cmdand dquote>             'report_props': {k: (list(v) if isinstance(v, list) else v) for k, v in rec['report_props'].items(
)}
cmdand dquote>         }, indent=2, default=str))
cmdand dquote>         print('---')
cmdand dquote> client.close()
cmdand dquote> "
{
  "category": "GroundwaterQualityRestrictedUseZone",
  "report_props": {
    "hasLink": [
      "https://drapubcdnprd.azureedge.net/ibis/attachments/gqruz/0007000248/GQRUZ_map_0007000248.pdf"
    ],
    "uri": "http://example.org/data/vic-unearthed/groundwater_report/7000248"
  }
}
---
{
  "category": "GroundwaterQualityRestrictedUseZone",
  "report_props": {
    "hasLink": [
      "https://drapubcdnprd.azureedge.net/ibis/attachments/gqruz/0007000073/GQRUZ_map_0007000073.pdf"
    ],
    "uri": "http://example.org/data/vic-unearthed/groundwater_report/7000073"
  }
}
---
{
  "category": "GroundwaterQualityRestrictedUseZone",
  "report_props": {
    "hasLink": [
      "https://drapubcdnprd.azureedge.net/ibis/attachments/gqruz/0007000096/GQRUZ_map_0007000096.pdf"
    ],
    "uri": "http://example.org/data/vic-unearthed/groundwater_report/7000096"
  }
}
---
{
  "category": "GroundwaterQualityRestrictedUseZone",
  "report_props": {
    "hasLink": [
      "https://drapubcdnprd.azureedge.net/ibis/attachments/gqruz/0007000118/GQRUZ_map_0007000118.pdf"
    ],
    "uri": "http://example.org/data/vic-unearthed/groundwater_report/7000118"
  }
}
---
{
  "category": "GroundwaterQualityRestrictedUseZone",
  "report_props": {
    "hasLink": [
      "https://drapubcdnprd.azureedge.net/ibis/attachments/gqruz/0007000158/GQRUZ_map_0007000158.pdf"
    ],
    "uri": "http://example.org/data/vic-unearthed/groundwater_report/7000158"
  }
}
---
{
  "category": "GroundwaterQualityRestrictedUseZone",
  "report_props": {
    "hasLink": [
      "https://drapubcdnprd.azureedge.net/ibis/attachments/gqruz/0007000175/GQRUZ_map_0007000175.pdf"
    ],
    "uri": "http://example.org/data/vic-unearthed/groundwater_report/7000175"
  }
}
---
```









======

(.venv) (.venv) veris-chat $ source .venv/bin/activate && python -c "
cmdand dquote> from rag_core.kg.client import get_kg_client
cmdand dquote> from rag_core.kg.context import format_parcel_context
cmdand dquote> 
cmdand dquote> client = get_kg_client()
cmdand dquote> context = client.get_parcel_context('210470171')
cmdand dquote> print(format_parcel_context('210470171', context))
cmdand dquote> client.close()
cmdand dquote> "
## Parcel Context (PFI: 210470171)

### Environmental Audits
- Date: 2008-07-03 00:00:00 | [PDF](https://drapubcdnprd.azureedge.net/ibis/attachments/envaudit/53X/62760-1/62670-1_a.pdf)

### EPA Licences
- No data found for this parcel

### Preliminary Risk Screening Assessments (PRSA)
- No data found for this parcel

### Priority Site Register
- No data found for this parcel

### Landfill Register (VLR, 500m buffer)
- No data found for this parcel

### Planning Overlays
- DESIGN AND DEVELOPMENT OVERLAY - SCHEDULE 12 (2017-06-07 00:00:00)
- PARKING OVERLAY - PRECINCT 7 (2017-06-07 00:00:00)
- DEVELOPMENT PLAN OVERLAY - SCHEDULE 4 (2017-06-07 00:00:00)
- PARKING OVERLAY - PRECINCT 7 (2017-06-07 00:00:00)
- DESIGN AND DEVELOPMENT OVERLAY - SCHEDULE 51 (2017-06-07 00:00:00)
- DESIGN AND DEVELOPMENT OVERLAY - SCHEDULE 51 (AREA 4) (2017-06-07 00:00:00)
- DESIGN AND DEVELOPMENT OVERLAY - SCHEDULE 1 (2021-09-27 00:00:00)
- PARKING OVERLAY - PRECINCT 1 (2018-09-18 00:00:00)
- DESIGN AND DEVELOPMENT OVERLAY - SCHEDULE 10 (2020-12-04 00:00:00)

### Historical Business Listings
- Agents - Tourists — Unknown [1896-01-01 00:00:00]
- Service Stations — Service stations/fuel storage (HIGH contamination risk) [1965-01-01 00:00:00]

### Groundwater Quality Restricted Use Zones (GQRUZ)
- Restricted use: Drinking water;Irrigation of crops (including domestic gardens) and parks;Livestock water supply;Water used for recreational purposes (e.g. swimming) | [PDF](https://drapubcdnprd.azureedge.net/ibis/attachments/gqruz/0007000096/GQRUZ_map_0007000096.pdf)
- Restricted use: Drinking water;Livestock water supply;Water used for recreational purposes (e.g. swimming) | [PDF](https://drapubcdnprd.azureedge.net/ibis/attachments/gqruz/0007000158/GQRUZ_map_0007000158.pdf)
- Restricted use: Drinking water;Water used for recreational purposes (e.g. swimming) | [PDF](https://drapubcdnprd.azureedge.net/ibis/attachments/gqruz/0007000248/GQRUZ_map_0007000248.pdf)
- Restricted use: Drinking water;Water used for recreational purposes (e.g. swimming) | [PDF](https://drapubcdnprd.azureedge.net/ibis/attachments/gqruz/0007000175/GQRUZ_map_0007000175.pdf)
- Restricted use: Drinking water;Livestock water supply;Water used for recreational purposes (e.g. swimming) | [PDF](https://drapubcdnprd.azureedge.net/ibis/attachments/gqruz/0007000118/GQRUZ_map_0007000118.pdf)
- Restricted use: Drinking water;Irrigation of crops (including domestic gardens) and parks;Livestock water supply;Water used for recreational purposes (e.g. swimming) | [PDF](https://drapubcdnprd.azureedge.net/ibis/attachments/gqruz/0007000073/GQRUZ_map_0007000073.pdf)
====