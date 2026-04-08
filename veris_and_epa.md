
Here's the project picture:

**Veris** is a private environmental consulting firm in Victoria, Australia. They do contamination assessments, soil/groundwater investigations, and prepare technical reports for land development projects.

**This project (veris-chat)** is a document-grounded conversational AI system built for Veris consultants (or their clients). The workflow:

1. A user starts a session and attaches PDF documents — these are statutory documents like environmental audit reports, EPA certificates, contamination assessments (e.g., `OL000071228 - Statutory Document.pdf` in your test data)

2. The system ingests, chunks, and embeds these PDFs into a session-scoped vector store

3. The user can then ask questions about the documents in natural language, e.g., "Is this site a priority site?", "What is the contamination status at 322 New Street, Brighton?"

4. The system retrieves relevant chunks, generates answers with inline citations pointing back to specific pages/documents, and maintains conversation memory across turns

**In short**: It's a RAG chatbot that lets Veris staff quickly query and cross-reference environmental/statutory documents instead of manually reading through dense PDF reports. The citations ensure traceability back to source documents — important in a regulatory context where you need to know exactly which document and page a conclusion comes from.

## Statutory Environmental Audit — Project Level (Victoria)

A statutory Environmental Audit is **not required because you engage Veris**.
It is required **only if triggered by land conditions or regulatory requirements**.

### When is an Environmental Audit required?

In Victoria, a statutory audit may be required if:

* The land is potentially contaminated
* The proposed use is a **sensitive use** (e.g. residential, school, childcare, aged care)
* The local Council imposes it as a planning condition
* EPA Victoria issues a direction

Only an **EPA-appointed independent Environmental Auditor** can issue a:

* Certificate of Environmental Audit, or
* Statement of Environmental Audit

### Veris’ role

Veris is a private consulting firm. They may:

* Conduct Phase 1 and Phase 2 contamination assessments
* Undertake soil and groundwater investigations
* Prepare technical and planning reports
* Coordinate with the EPA-appointed auditor

However:

* Veris cannot issue a statutory Environmental Audit certificate
* Veris is not a regulator

### Bottom line

An Environmental Audit is required only when mandated by planning or environmental regulation.
Veris can support the process, but the formal legal audit outcome must be issued by an EPA-appointed Environmental Auditor.
