# AssureX Claim Engine

### Multimodal Consumer Electronics Warranty Adjudication Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> AssureX is an enterprise-grade multimodal warranty claims processing engine that automates consumer electronics warranty adjudication. It combines a **Python Tabular Random Forest Classifier**, a **Google Teachable Machine (GTM) Visual Classifier**, a **Deterministic Warranty Rule Engine** (12 policy rules, including PTA DIRBS / FBR NTN compliance for the Pakistani market), and a **Multimodal Arbitration Decision Engine** to deliver transparent, auditable claim decisions.

**Reported benchmark results** (per the project's engineering report — see [Note on Benchmark Figures](#note-on-benchmark-figures)): **92.0%** tabular-model test accuracy, **88.9%** cross-model agreement, and **100.0%** adjudication accuracy on a 36-claim unseen evaluation set, with a target end-to-end pipeline latency of under 500 ms per claim.

**License:** MIT (as declared in the project's own README badge; no `LICENSE` file was included among the reviewed project files — see [License](#license)).

---

> ### ⚠️ Important Note on Source Material
> This README was produced from the project's **documentation set** — `README.md`, `PROJECT_REPORT.md`, `TECHNICAL_BLOG_POST.md`, `DEMO_VIDEO_SCRIPT.md`, `data_dictionary.md`, `AI_USAGE.md`, and `requirements.txt` — rather than from a direct inspection of the application's source code (no `backend/`, `frontend/`, or model-artifact files were provided alongside the documentation). Every statement below reflects what these documents describe. Where the documents disagree with one another, or where a claim cannot be cross-checked against `requirements.txt` (the one concrete, machine-verifiable artifact provided), this is called out explicitly rather than silently resolved — see [Note on Benchmark Figures](#note-on-benchmark-figures) and [Discrepancy: ML/OCR Dependencies vs. requirements.txt](#discrepancy-mlocr-dependencies-vs-requirementstxt).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Requirements](#3-system-requirements)
4. [Technology Stack](#4-technology-stack)
5. [Installation & Setup Guide](#5-installation--setup-guide)
6. [Environment Configuration](#6-environment-configuration)
7. [Database Setup & Initialization](#7-database-setup--initialization)
8. [Running the Application](#8-running-the-application)
9. [Authentication & Authorization](#9-authentication--authorization)
10. [Complete Feature Execution Guide](#10-complete-feature-execution-guide)
11. [Dashboard & Analytics](#11-dashboard--analytics)
12. [API Documentation](#12-api-documentation)
13. [Repository Structure](#13-repository-structure)
14. [Database Design & Data Dictionary](#14-database-design--data-dictionary)
15. [Machine Learning & Decision Engine](#15-machine-learning--decision-engine)
16. [Testing](#16-testing)
17. [Security](#17-security)
18. [Error Handling & Validation](#18-error-handling--validation)
19. [Reports, Export & Documents](#19-reports-export--documents)
20. [Demo / Default Credentials](#20-demo--default-credentials)
21. [Troubleshooting & FAQ](#21-troubleshooting--faq)
22. [Deployment Guide](#22-deployment-guide)
23. [Performance & Architecture](#23-performance--architecture)
24. [System Limitations](#24-system-limitations)
25. [Future Improvements](#25-future-improvements)
26. [License](#26-license)
27. [Developer / Credits](#27-developer--credits)

---

## 1. Project Overview

### What the system does

AssureX is a full-stack warranty claims platform for **consumer electronics and home appliances** (smartphones, laptops, washing machines, smart TVs, and audio/soundbars). It takes a claimant through product/warranty registration and a guided claim-filing wizard, then automatically adjudicates each claim by combining:

- A **tabular machine learning model** (Random Forest) trained on structured claim attributes,
- A **computer vision model** (a Google Teachable Machine–trained MobileNetV2 classifier) that evaluates a standardized visual "Claim Summary Card,"
- A **deterministic rule engine** enforcing 12 hard warranty-policy and regulatory rules, and
- A **decision arbitration engine** that reconciles the two models' outputs against the rule engine's verdict and produces a final, explainable decision.

### Main purpose and objectives

Per the project's own stated objectives (`PROJECT_REPORT.md`), AssureX aims to:
- Automate a large share of routine, "clean" warranty claims without human review,
- Keep the two ML models in agreement on unseen data at a high rate,
- Achieve strong adjudication accuracy on balanced benchmark data,
- **Never** auto-approve a claim that fails a hard policy rule (expired warranty, excluded damage, regulatory non-compliance), regardless of how confident the ML models are, and
- Maintain a complete, immutable audit trail of every claim event.

### Problems it solves

The project documentation frames this as a response to five industry pain points: slow manual claim turnaround, warranty fraud/laundering, regulatory non-compliance (specifically Pakistan's PTA DIRBS handset-blacklist regime and FBR NTN/STRN tax-invoice requirements), inconsistent human adjudication, and the lack of tooling that combines tabular, visual, and rule-based evidence into one decision.

### Target users / businesses

- **Customers** — claim filers and product owners.
- **Service Centers** — staff performing physical diagnostics and logging repair history.
- **Claim Reviewers** — underwriters who adjudicate claims flagged for manual review.
- **Administrators** — operators monitoring system-wide KPIs, model performance, and audit data.

### Major capabilities (at a glance)

| Capability | Description |
|---|---|
| Product & warranty registration | Customers register a product; the system auto-provisions a matching warranty record. |
| 4-step claim wizard with OCR | Guided claim submission with live receipt OCR field extraction. |
| Dual-model + rule-based adjudication | Tabular ML + visual ML + 12-rule policy engine, reconciled by a decision engine. |
| Manual review workflow | Reviewer queue with approve / reject / override actions and mandatory notes. |
| Executive analytics dashboard | Chart.js-based KPI and trend visualizations for administrators. |
| PDF certificate generation | ReportLab-generated, downloadable claim adjudication certificates. |
| CSV data export | Filtered claim data export for external analysis. |
| Batch model-comparison pipeline | Script to evaluate 30+ (or all 225) unseen test claims and produce comparison reports. |
| Immutable audit logging | Every submission, prediction, status change, and reviewer action is logged. |

---

## 2. Key Features

The following features are described consistently across `README.md`, `PROJECT_REPORT.md`, and `AI_USAGE.md`. Each is presented as implemented; none of the underlying source files were available for direct code verification (see the note at the top of this document).

1. **Role-Based User Authentication** — Registration/login for four roles (`customer`, `service_center`, `claim_reviewer`, `admin`) using bcrypt-style password hashing and JWT bearer tokens, with no demo-bypass shortcuts (per `AI_USAGE.md`, Entry 7).
2. **Product & Warranty Registration** — Customers register a product (category, brand, model, serial number, purchase price/date, retailer); the system computes and stores a corresponding warranty record (default 12-month duration).
3. **4-Step Claim Submission Wizard** — Product selection → fault/damage declaration → document evidence upload → extracted-data verification and submission.
4. **Live Receipt OCR** — Extraction of purchase date, price, retailer, invoice number, and serial number from uploaded receipt images, with per-field confidence flags and regex patterns tuned for Pakistani NTN/STRN tax-number formats.
5. **SHA-256 Duplicate Document Detection** — Every uploaded file is hashed on ingestion; identical hashes across claims are flagged as a potential fraud indicator.
6. **Dual Machine-Learning Classification** — A tabular Random Forest classifier and a separate visual (GTM/MobileNetV2) classifier each independently score a claim as Valid / Invalid / Manual Review.
7. **Deterministic 12-Rule Policy Engine** — Category-specific rules covering excluded damage, unauthorized repair, warranty-active status, proof-of-purchase presence, serial matching (with PTA DIRBS compliance for phones), mandatory-document completeness, fault coverage, reporting-period validity, lemon-law thresholds, prior-replacement checks, duplicate-claim detection, and chronological contradiction detection.
8. **Multimodal Decision Arbitration** — Computes the confidence gap between the two ML models, classifies their agreement level (Strong / Acceptable / Weak Match, Disagreement, or Uncertain), applies rule-engine vetoes, and produces a final decision with a natural-language explanation (supporting factors, opposing factors, missing evidence).
9. **Manual Review Queue & Reviewer Actions** — Reviewers can inspect the multi-model consensus matrix and approve, reject, or override a claim with mandatory justification notes.
10. **Executive Admin Dashboard** — Chart.js-based visualizations of claim volume, status distribution, model agreement rate, and category breakdown.
11. **Search & Multi-Criteria Filtering** — Filter claims by status, product category, and free-text keyword (claim ID, serial number, customer name, fault keywords).
12. **PDF Claim Certificates** — ReportLab-generated downloadable adjudication certificates including claimant/product/serial metadata, warranty timeline, the 12-rule audit checklist, and the multimodal decision breakdown.
13. **CSV Export** — Filtered claim data export (claim attributes, predictions, confidences, rule-audit results, reviewer notes) for external actuarial/analytical use.
14. **Batch Unseen-Claims Evaluation Pipeline** — A standalone script (`run_model_comparison_pipeline.py`) that runs a configurable number of unseen test claims through the full pipeline and emits CSV/Markdown comparison reports.
15. **Immutable Audit Trail** — Every claim submission, prediction run, status transition, and reviewer decision is written to an `audit_logs` table with timestamp, actor, and JSON detail payload.
16. **Automated Test Suites** — Pytest suites covering authentication, claims lifecycle, rule engine, decision engine, RBAC/security, OCR/duplicate detection, and model inference, plus standalone end-to-end auth/RBAC verification scripts.

---

## 3. System Requirements

As stated in the project's own README/`PROJECT_REPORT.md`:

| Requirement | Specification |
|---|---|
| **Operating Systems** | Windows 10/11, macOS 13+, Ubuntu 20.04/22.04 LTS |
| **Python Runtime** | Python 3.10.x through 3.14.x (64-bit recommended) |
| **Package Manager** | pip 22.0+ |
| **Disk Space** | Minimum 2 GB free (dataset, model weights, OCR libraries) |
| **Memory** | Minimum 4 GB RAM; 8 GB recommended for local OCR inference |
| **Database** | SQLite by default (`database/assurex.db`), documented as "PostgreSQL compatible" via `DATABASE_URL` — no PostgreSQL-specific driver (e.g., `psycopg2`) appears in `requirements.txt`, so PostgreSQL usage would require adding that dependency separately |

No explicit GPU requirement is documented; the visual (GTM) model is described as CPU-inferable via TensorFlow/Keras, with a documented fallback path for platforms where native TensorFlow C++ components are unavailable (see [System Limitations](#24-system-limitations) and [Troubleshooting](#21-troubleshooting--faq)).

---

## 4. Technology Stack

### Backend (confirmed via `requirements.txt`)

| Component | Library | Version Constraint |
|---|---|---|
| Web framework | FastAPI | `>=0.115.0` |
| ASGI server | Uvicorn | `>=0.30.0` |
| ASGI toolkit | Starlette | `>=0.38.0` |
| ORM / database toolkit | SQLAlchemy | `>=2.0.0` |
| Data validation | Pydantic | `>=2.8.0` |
| Authentication tokens | PyJWT | `>=2.9.0` |
| Multipart form handling | python-multipart | `>=0.0.9` |
| Email validation | email-validator | `>=2.0.0` |
| HTTP client | httpx | `>=0.27.0` |
| PDF generation | ReportLab | `>=4.2.0` |
| Image handling | Pillow | `>=10.0.0` |

### Machine Learning / Data (confirmed via `requirements.txt`)

| Component | Library | Version Constraint |
|---|---|---|
| Data manipulation | pandas | `>=2.0.0` |
| Numerical computing | NumPy | `>=1.24.0` |
| Classical ML | scikit-learn | `>=1.3.0` |
| Model serialization | joblib | `>=1.3.0` |
| Date utilities | python-dateutil | `>=2.8.0` |

### Frontend (per `README.md` / `PROJECT_REPORT.md` / `AI_USAGE.md`)

| Component | Technology |
|---|---|
| Structure | HTML5 single-page application (`frontend/index.html`) |
| Styling | Custom CSS (`frontend/css/style.css`) plus Bootstrap 5.3 |
| Scripting | Vanilla JavaScript (ES6+), no SPA framework (`frontend/js/app.js`) |
| Charts | Chart.js 4.4 |

### Database

- **Default:** SQLite (`database/assurex.db`), auto-initialized on first run.
- **Documented alternative:** PostgreSQL, via the `DATABASE_URL` environment variable — noted as "compatible" in the environment configuration, but **no PostgreSQL driver is listed in `requirements.txt`**, so this path is not verified as installable out of the box.

### Authentication & Security (as documented)

- JWT bearer tokens (HMAC-SHA256, 24-hour expiry by default), issued and verified using **PyJWT** (the package actually present in `requirements.txt`; the project report separately mentions `python-jose`, but only PyJWT appears in the dependency manifest — see the discrepancy note below).
- Password hashing described throughout the documentation as "salted bcrypt," though **no bcrypt/passlib package appears in `requirements.txt`** — see [Discrepancy: ML/OCR Dependencies vs. requirements.txt](#discrepancy-mlocr-dependencies-vs-requirementstxt).
- Role-Based Access Control (RBAC) via a `require_role()` dependency-injection gatekeeper (per `PROJECT_REPORT.md` §10.6).

### Third-Party Services / APIs

None. The documentation explicitly states that **live integration with PTA DIRBS telecom carrier APIs is out of scope** and is instead simulated via the deterministic rule engine's own compliance checks.

### Machine Learning / AI Technologies

- **Tabular model:** `RandomForestClassifier` (scikit-learn), trained on 19 engineered features, serialized with `joblib` to `model/claim_classifier.pkl`.
- **Visual model:** A Google Teachable Machine–exported MobileNetV2 Keras model (`model/gtm_model/keras_model.h5`), described as requiring TensorFlow/Keras at inference time.

#### Discrepancy: ML/OCR Dependencies vs. `requirements.txt`

The project's documentation (`README.md`, `PROJECT_REPORT.md`, `AI_USAGE.md`) repeatedly and specifically describes two components that depend on packages **not present in the provided `requirements.txt`**:

- **The GTM visual classifier** is described as loading a Keras `.h5` model and depending on **TensorFlow** (`PROJECT_REPORT.md` §6 lists `tensorflow` 2.15+ as a dependency) — but `tensorflow` and `keras` are absent from `requirements.txt`.
- **The receipt OCR pipeline** (`receipt_ocr.py`) is described as being built on **EasyOCR** — but `easyocr` (and no alternative such as `pytesseract`) appears in `requirements.txt`.
- Password hashing is described as **bcrypt**-based, but neither `bcrypt` nor `passlib` appears in `requirements.txt`.

This README does not assume these libraries are actually installed or importable in a fresh `pip install -r requirements.txt` environment, since the manifest does not confirm it. Two explanations are possible: (a) these packages are installed separately/manually and were simply omitted from `requirements.txt`, or (b) the GTM/OCR/bcrypt code paths have documented fallback behavior not fully captured in the provided files (the troubleshooting notes do describe a "TensorFlow C++ DLL" fallback for the GTM classifier specifically). **This could not be verified from the files provided**, and a developer setting up the project from `requirements.txt` alone should expect to additionally install `tensorflow`, `easyocr` (or confirm the documented fallback path), and a bcrypt-compatible hashing library if these features are required.

---

## 5. Installation & Setup Guide

The following steps are transcribed directly from the project's own installation instructions.

### Step 1 — Obtain the Repository

```bash
# Open a terminal in the project directory
cd /path/to/AssureX
```
*(The original documentation shows a Windows-specific example path, `cd c:\Users\omar\Desktop\Code\AssureX`; use the actual path to your copy of the project.)*

### Step 2 — Create and Activate a Python Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Upgrade pip and Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> See [Discrepancy: ML/OCR Dependencies vs. requirements.txt](#discrepancy-mlocr-dependencies-vs-requirementstxt) — if you intend to use the visual (GTM) classifier or the OCR pipeline, verify separately whether `tensorflow` / `easyocr` need to be installed manually in your environment, as they are not listed in `requirements.txt`.

### Step 4 — Verify Model Artifact Placement

The documentation states that trained model weights are **pre-bundled in the repository** and must reside at these exact paths:

```
AssureX/
├── model/
│   ├── claim_classifier.pkl         # Trained Random Forest Tabular Model (scikit-learn)
│   ├── preprocessing.pkl            # StandardScaler, LabelEncoder, and feature schemas
│   └── gtm_model/
│       ├── keras_model.h5           # Google Teachable Machine MobileNetV2 Vision Model
│       └── labels.txt               # Class label index mappings (Valid, Invalid, Manual Review)
```

### Step 5 — Configure the Environment (Optional)

See [Environment Configuration](#6-environment-configuration) below. The application is documented to run with sensible defaults if no `.env` file is created.

### Step 6 — Initialize and Seed the Database

```bash
python seed_db.py
```

See [Database Setup & Initialization](#7-database-setup--initialization) for expected output.

### Step 7 — Run the Application

See [Running the Application](#8-running-the-application) below.

### Step 8 — (Optional) Run the Test Suites

See [Testing](#16-testing) below.

---

## 6. Environment Configuration

The documentation states that AssureX "uses sensible defaults and automatically initializes an SQLite database in `database/assurex.db`." An **optional** `.env` file may be created in the project root to customize the database connection, cryptographic secrets, or token lifetime. The example configuration provided in the project's own documentation is reproduced below — **the values shown are the documentation's own illustrative example, not verified live secrets**, and should still be replaced with your own generated values in any real deployment:

```ini
# .env.example — copy to .env and adjust as needed

# Database Connection (SQLite default; PostgreSQL "compatible" per docs,
# though no PostgreSQL driver is listed in requirements.txt — see Section 4)
DATABASE_URL=sqlite:///database/assurex.db

# Cryptographic Security
# ⚠️ CHANGE THIS: generate a strong, random secret for any non-local deployment.
# Do not reuse the illustrative value shown in the project's own documentation.
JWT_SECRET_KEY=change-this-to-a-long-random-secret-value
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Logging & Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
```

| Variable | Purpose | Must Be Changed By Developer? |
|---|---|---|
| `DATABASE_URL` | Connection string for the application database | Only if moving off the default SQLite file |
| `JWT_SECRET_KEY` | Signing secret for JWT bearer tokens | **Yes**, for any deployment beyond local evaluation |
| `JWT_ALGORITHM` | JWT signing algorithm | Only if changing the signing scheme |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token lifetime in minutes (default: 1440 = 24 hours) | Optional |
| `ENVIRONMENT` | Application environment flag | Optional |
| `LOG_LEVEL` | Logging verbosity | Optional |

No other environment variables (API keys, third-party service credentials, etc.) are documented anywhere in the provided files.

---

## 7. Database Setup & Initialization

**Database engine:** SQLite by default, auto-created at `database/assurex.db` on first run — no manual schema migration step is documented as being required beyond running the seed script below.

**Seed script:**

```bash
python seed_db.py
```

Per the documentation, this creates the schema and populates:
- One demo user per role (`admin`, `reviewer`, `service_center`, `customer`),
- Registered products and active warranties for the demo customer,
- Prior repair histories, and
- **11 to 12 distinct demo claim scenarios** (the README states "11 distinct demo claim scenarios" while the sample console output and `AI_USAGE.md` reference "12 comprehensive demo claims" — this minor count discrepancy exists in the source documentation itself and could not be resolved without the actual seed script).

**Documented expected console output:**

```
======================================================================
 ASSUREX CLAIMS PROCESSING ENGINE — DATABASE SEEDING
======================================================================
[*] Database URL: sqlite:///.../database/assurex.db
[+] User created: admin (Role: admin)
[+] User created: reviewer (Role: claim_reviewer)
[+] User created: service_center (Role: service_center)
[+] User created: customer (Role: customer)
[+] Seeded 12 comprehensive demo claims with documents and predictions
[+] Database seeding completed successfully!
```

**Resetting the database:** delete `database/assurex.db` and re-run `python seed_db.py`.

For the full relational schema populated by this process, see [Database Design & Data Dictionary](#14-database-design--data-dictionary).

---

## 8. Running the Application

### Development Mode

**Option A — Launcher script (documented as recommended):**
```bash
python run_server.py
```

**Option B — Direct Uvicorn invocation:**
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

The provided documentation does not describe a distinct, separately-configured production launch command (e.g., a Gunicorn/Uvicorn-worker process manager configuration); it presents the same launcher/Uvicorn commands above as the way to run the application, with `ENVIRONMENT=production` settable via `.env`. Treat any production hardening (process management, reverse proxy, TLS termination, disabling `--reload`) as **not explicitly documented** and something to configure yourself — see [Deployment Guide](#22-deployment-guide).

### URLs (once running)

| Endpoint | URL | Purpose |
|---|---|---|
| Web Application UI | `http://localhost:8000/` | Main SPA (all roles) |
| Interactive Swagger API Docs | `http://localhost:8000/docs` | Auto-generated OpenAPI/Swagger UI |
| ReDoc API Documentation | `http://localhost:8000/redoc` | Alternative auto-generated API reference |

To use a different port, the documentation references `python run_server.py --port 8080` (see [Troubleshooting](#21-troubleshooting--faq)).

---

## 9. Authentication & Authorization

### Mechanism

- **Registration/Login:** Username- or email-based login with a password, documented as enforcing "strict cryptographic password verification" with **no demo bypass or one-click shortcuts** (`README.md` §8; corroborated by the security-audit entry in `AI_USAGE.md`, Entry 7).
- **Tokens:** JWT bearer tokens, HMAC-SHA256 signed, with a documented 24-hour default expiry (`ACCESS_TOKEN_EXPIRE_MINUTES=1440`). Tokens are transported via the standard `Authorization: Bearer <token>` header (per `AI_USAGE.md`, Entry 8, which also documents a fix adding a 60-second decode leeway for clock-drift tolerance).
- **Password storage:** Documented as salted bcrypt hashing — **note the `requirements.txt` discrepancy flagged in Section 4**.

### User Roles & Access Scope

| Role | Username | Access Scope (as documented) |
|---|---|---|
| System Administrator | `admin` | System telemetry, model-agreement KPIs, CSV export, audit logs |
| Claim Reviewer | `reviewer` | Manual review queue, evidence inspection, approve/reject/override actions |
| Service Center | `service_center` | Physical diagnostic intake, repair logging, serial audits |
| Customer | `customer` | Product registration, claim filing wizard, OCR verification, status tracking |

### Role-Based Access Control (RBAC)

- Enforced server-side via a `require_role()` dependency-injection gatekeeper described in `backend/auth.py` (`PROJECT_REPORT.md` §10.6).
- Verified, per `AI_USAGE.md`, by a dedicated script (`verify_all_roles_rbac.py`) that confirms unauthenticated and cross-role requests are rejected with HTTP `401`/`403`.
- The frontend is described as adapting its navigation and visible actions dynamically based on the authenticated user's role.

### Protected Routes / Pages

Route-level protection is described at the API layer (role-gated FastAPI dependencies on the `/reviews`, `/admin`, and `/export` routers in particular, based on the role-scope table above); the frontend correspondingly hides navigation entries a given role should not use. The specific list of which frontend routes are hard-blocked versus merely hidden in the UI **could not be verified** without the `frontend/js/app.js` source.

---

## 10. Complete Feature Execution Guide

This walkthrough is reproduced from the project's own documented feature-execution guide (`README.md` §9), which doubles as the project's evaluation script.

### Feature 1: User Authentication & Role Switching
- **Access:** `http://localhost:8000/`
- **Workflow:** Sign in with the Customer credentials → observe the navigation change to show the Customer Dashboard (Active Warranties, Filed Claims, Register Product, File Claim) → open the profile badge to view/update legal name and password → log out and sign in as `reviewer` or `admin` to confirm role-appropriate navigation changes.
- **Expected result:** Navigation and available actions differ per role; no role can access another role's dashboard sections.

### Feature 2: Product & Warranty Registration
- **Access:** "Register Product" in the navigation bar (Customer role).
- **Workflow:** Fill in product category (Smartphone / Laptop / Washing Machine / Smart TV / Audio-Soundbar), marketing name, brand & model, chassis serial number, purchase date & retailer, purchase price → submit.
- **Expected result:** A `Product` entity is created and a corresponding warranty record (default 12-month duration) is automatically calculated and shown on the dashboard.

### Feature 3: 4-Step Claim Submission Wizard with Live OCR
- **Access:** "File New Claim" from the Customer Dashboard.
- **Workflow:**
  1. **Select Product** — choose a registered, active-warranty product.
  2. **Fault & Damage Declarations** — fault occurrence date, damage category, detailed fault description, prior-repair/replacement declarations.
  3. **Document Evidence Upload** — purchase receipt/tax invoice, dealer-stamped warranty card, chassis serial barcode evidence; the OCR pipeline extracts receipt date, price, and retailer NTN.
  4. **Verify & Submit** — review extracted fields/confidence indicators, confirm the declaration checkbox, submit.
- **Expected result:** The system runs the full multimodal pipeline and assigns a tracking number (e.g., `CLM-2026-00014`).
- **Important business rule:** Only products with an active, registered warranty can have a claim filed against them.

### Feature 4: Claim Status Tracking & Visual Timeline
- **Access:** "Track Claim" in the navigation bar.
- **Workflow:** Enter a Claim ID.
- **Expected result:** A live status badge (Approved / Rejected / Manual Review Required), a visual milestone timeline (Submitted → Automated Multi-Model Evaluation → Adjudication Decision), a warranty coverage card, and an evidence gallery with SHA-256 checksum verification.

### Feature 5: Reviewer Dashboard & Manual Review Queue
- **Access:** "Reviewer Queue" (Claim Reviewer role).
- **Workflow:** Review claims flagged Manual Review Required → open a claim → inspect the Multi-Model Consensus Matrix (Tabular vs. Visual vs. Rule Engine) and the automated decision explanation → choose Approve / Reject / Override, providing mandatory notes for an override.
- **Expected result:** Claim status updates immediately; the decision is written to the immutable audit trail.

### Feature 6: Executive Admin Analytics & Chart.js Visualizations
- **Access:** "Admin Dashboard" (Administrator role).
- **Workflow:** Review Total Claims Filed, Adjudication Distribution (Valid/Invalid/Manual Review), Multi-Model Agreement Rate, and Average Top Confidence; inspect the Chart.js trend and category-distribution charts.

### Feature 7: Search, Multi-Criteria Filter & Audit Querying
- **Access:** Search & Filter toolbar on the Admin Dashboard.
- **Workflow:** Filter by status, product category, or free-text keyword (claim ID, serial number, customer name, fault keywords). Table updates without a page reload.

### Feature 8: Downloadable PDF Claim Certificates (ReportLab)
- **Access:** "Download Claim Report (PDF)" button in the claim tracker (or Reviewer/Admin claim view).
- **Expected result:** A PDF certificate (e.g., `CLM-2026-DEMO-001_Report.pdf`) containing a branded header with tracking barcode, claimant/product/serial verification metadata, warranty coverage timeline, the 12-rule audit checklist, and the full multimodal decision breakdown.

### Feature 9: Filtered Claims CSV Data Export
- **Access:** "Export CSV" button on the Admin Dashboard.
- **Workflow:** Apply desired filters (or none, for all records) → click Export CSV.
- **Expected result:** A downloaded CSV containing claim attributes, predictions, confidences, rule-audit results, and reviewer notes.

### Feature 10: Automated Batch Unseen Claims Evaluation Pipeline
```bash
# Evaluate the default 36 unseen test claims from dataset/test.csv
python run_model_comparison_pipeline.py

# Evaluate all 225 unseen claims in the test set
python run_model_comparison_pipeline.py --all
```
- **Outputs:** `reports/unseen_test_claims_report.csv` and `reports/unseen_test_claims_report.md`.

---

## 11. Dashboard & Analytics

The Administrator dashboard is documented to provide:

- **KPIs:** Total claims filed, Valid/Invalid/Manual-Review adjudication distribution, multi-model agreement rate (Python tabular vs. GTM visual), and average top-confidence telemetry across active predictions.
- **Charts (Chart.js):**
  - Claim Adjudication Trends — bar/pie distribution of claim outcomes.
  - Category Distribution — claim volume across Smartphones, Laptops, Appliances, and Audio.
- **Filters:** Status, product category, and free-text keyword search, applied live without page reload (see Feature 7 above).
- **Search:** Keyword search across claim IDs, serial numbers, customer names, and fault-description keywords.
- **Export:** Filtered-data CSV export (see [Reports, Export & Documents](#19-reports-export--documents)).

No dedicated customer- or reviewer-level analytics screens beyond the claim tracker and reviewer queue are described in the provided documentation.

---

## 12. API Documentation

### Architecture

A modular FastAPI application (`backend/main.py`) mounting a set of route routers, each scoped to one functional domain, secured with JWT bearer-token authentication and role-gated dependencies. Interactive documentation is auto-generated by FastAPI/Swagger at `/docs` and ReDoc at `/redoc` — these are the authoritative, always-current source for exact request/response schemas, since the underlying route source files were not included among the reviewed documents.

### Documented Route Groups

| Router | Responsibility (as described in `PROJECT_REPORT.md` §10.6) |
|---|---|
| `/auth` | Registration, login, token issuance/refresh, current-user profile |
| `/products` | Product registration, serial validation, per-user inventory |
| `/warranties` | Warranty status lookups, policy inquiries |
| `/claims` | Claim submission, evidence upload, OCR verification trigger, classification execution, PDF report generation |
| `/reviews` | Manual review queue retrieval, approval/rejection/override submission |
| `/admin` | Dashboard statistics, confusion-matrix/agreement metrics, audit-log inspection |
| `/export` | Filtered CSV export streaming |

### Specific Endpoints Referenced in the Documentation

- `POST /claims/` — submit a claim with payload and evidence files (per the sequence diagram in `PROJECT_REPORT.md` §12.4).
- `GET /reviews/queue/` — retrieve claims pending manual review.
- `POST /reviews/adjudicate` — submit an approve/reject/override decision.
- `GET /api/claims/{claim_id}/report` — download the PDF claim certificate (per `AI_USAGE.md`, Entry 9).

### Authentication Requirements

All routes except `/auth/login` and `/auth/register` are documented as requiring a valid `Authorization: Bearer <JWT>` header; role-restricted routes additionally enforce `require_role()` checks (e.g., `/admin/*` and `/reviews/*` are restricted to `admin` and `claim_reviewer`/`admin` respectively, per the role-scope table in Section 9).

### Error Handling

Not itemized endpoint-by-endpoint in the provided documentation. Standard FastAPI/Pydantic behavior (automatic `422 Unprocessable Entity` on request-validation failure, framework-level `401`/`403` on auth/RBAC failure) is the expected baseline given the declared stack, but **specific custom error-response shapes were not confirmed** from the files provided — consult `/docs` on a running instance for authoritative, current schemas.

### Full Endpoint Reference

**Not fully enumerable from the provided documentation.** For a complete, accurate, and current list of every endpoint, HTTP method, and request/response schema, use the live interactive documentation once the application is running:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 13. Repository Structure

The directory tree below is reproduced from the project's own documentation (`README.md` §10 / `PROJECT_REPORT.md` §10), since the actual repository filesystem was not provided for direct inspection.

```text
AssureX/
├── backend/                         # FastAPI Application Core
│   ├── main.py                      # Application assembly, static mounting, CORS
│   ├── config.py                    # Database paths, JWT secret, storage config
│   ├── database.py                  # SQLAlchemy engine & session factory
│   ├── models.py                    # 9 relational entities (User, Product, Claim, etc.)
│   ├── auth.py                      # Password hashing, JWT creation & RBAC gatekeeper
│   ├── pipeline.py                  # Multimodal pipeline orchestrator
│   └── routes/                      # Modular RESTful API route controllers
│       ├── auth.py                  # Login, register, profile
│       ├── products.py              # Product inventory management
│       ├── warranties.py            # Warranty lookups
│       ├── claims.py                # Claim filing, OCR, PDF generation
│       ├── reviews.py               # Manual review queue & overrides
│       ├── admin.py                 # Telemetry, metrics, analytics
│       └── export.py                # CSV export streaming
├── frontend/                        # Responsive client SPA
│   ├── index.html                   # Single-page application markup
│   ├── css/style.css                # Custom styling & responsive layouts
│   └── js/app.js                    # Client application logic, API calls, Chart.js
├── model/                           # Trained machine learning weights
│   ├── claim_classifier.pkl         # Tuned Random Forest model
│   ├── preprocessing.pkl            # StandardScaler and encoding mappings
│   └── gtm_model/                   # Google Teachable Machine artifacts
│       ├── keras_model.h5           # Fine-tuned MobileNetV2 Keras model
│       └── labels.txt               # Class label mappings
├── policies/                        # Category policy configuration
│   ├── smartphone_policy.json       # PTA DIRBS rules, liquid-damage exclusions, screen terms
│   ├── laptop_policy.json           # General computing / consumer electronics policy
│   └── washing_machine_policy.json  # Domestic appliance & motor warranty rules
├── config/                          # Decision engine thresholds
│   └── thresholds.json              # Inter-model confidence gap & arbitration thresholds
├── dataset/                         # Benchmarking claims dataset (1,500 records)
│   ├── train.csv                    # 70% stratified training split
│   ├── val.csv                      # 15% stratified validation split
│   └── test.csv                     # 15% stratified unseen testing split
├── documentation/                   # Specifications & technical documentation
│   ├── PROJECT_REPORT.md            # Comprehensive project engineering report
│   └── data_dictionary.md           # Dataset data dictionary
├── sample_claims/                   # 11 dedicated demo test fixtures (JSON)
├── tests/                           # Pytest test suites
│   ├── conftest.py                  # Fixtures: TestClient, in-memory SQLite DB, seeded entities
│   ├── test_auth.py                 # Authentication, password hashing, token validation
│   ├── test_claims.py               # Claim submission, status transitions, PDF export
│   ├── test_decision_engine.py      # Arbitration matrix, consistency tiers, explanations
│   ├── test_rule_engine.py          # 12 policy rules, Pakistan PTA/NTN, lemon laws
│   ├── test_security_rbac.py        # Role privilege enforcement, SQLi/XSS/bypass blocks
│   ├── test_ocr_and_duplicates.py   # OCR parsing, SHA-256 duplicate detection
│   └── test_model_inference.py      # Tabular and GTM model inference guarantees
├── CREDENTIALS.md                   # Evaluation accounts & scenario guide
├── requirements.txt                 # Pinned project dependencies
├── run_server.py                    # Uvicorn server launcher script
├── seed_db.py                       # Database seeding script
├── gtm_classifier.py                # Teachable Machine visual classifier
├── predict_with_confidence.py       # Tabular ML inference module
├── receipt_ocr.py                   # Receipt OCR parsing module
├── decision_engine.py                # Multimodal arbitration decision engine
├── generate_claims_dataset.py       # Synthetic dataset generation pipeline
├── run_feature_engineering.py       # Feature engineering pipeline
├── run_model_comparison_pipeline.py # Unseen-claims batch evaluator
├── test_login_end_to_end.py         # Standalone end-to-end auth verification script
└── verify_all_roles_rbac.py         # Standalone RBAC verification script
```

### Purpose of Key Files

| File / Directory | Purpose |
|---|---|
| `backend/main.py` | Assembles the FastAPI app, mounts static assets, configures CORS |
| `backend/auth.py` | Password hashing, JWT issuance, and the `require_role()` RBAC dependency |
| `backend/pipeline.py` | Orchestrates the tabular → visual → rule → decision pipeline per claim |
| `decision_engine.py` | Implements `final_claim_decision()` — the arbitration logic described in Section 15 |
| `receipt_ocr.py` | Extracts structured fields from uploaded receipt images |
| `policies/*.json` | Declarative, per-category warranty policy rules consumed by the rule engine |
| `config/thresholds.json` | Externalized confidence-gap thresholds used by the decision engine |
| `seed_db.py` | Creates schema and populates demo users, products, warranties, and claims |
| `run_model_comparison_pipeline.py` | Batch-evaluates unseen claims and writes CSV/Markdown comparison reports |

---

## 14. Database Design & Data Dictionary

### Entity-Relationship Overview

The database consists of **9 normalized relational entities**, as documented in `PROJECT_REPORT.md` §11.1:

```
users ──1:N──▶ products ──1:1──▶ warranties ──1:N──▶ repair_histories
  │
  ├──1:N──▶ claims ──1:N──▶ documents
  │            │
  │            ├──1:1──▶ predictions
  │            └──1:N──▶ reviews
  │
  └──1:N──▶ audit_logs
```

### Table Summaries (full column-level detail is in `PROJECT_REPORT.md` §11.2)

| Table | Purpose | Key Columns |
|---|---|---|
| `users` | Authentication & role identity | `username`, `email`, `hashed_password`, `role` |
| `products` | Registered hardware inventory | `product_category`, `serial_number` (unique), `user_id` (FK) |
| `warranties` | Coverage record per product | `product_id` (FK, unique), `warranty_status`, `warranty_expiry_date` |
| `claims` | Central claim record | `claim_id` (public, e.g. `CLM-2026-00001`), `status`, `fault_occurrence_date` |
| `documents` | Uploaded evidence files | `sha256_hash`, `is_duplicate`, `duplicate_of_document_id` (FK) |
| `repair_histories` | Prior service records | `is_authorized`, `fault_repaired`, `repair_center` |
| `predictions` | One-to-one ML/rule result per claim | `python_prediction`, `gtm_prediction`, `rule_engine_result`, `final_decision`, `model_consistency_status`, `confidence_difference` (all JSON/text payloads) |
| `reviews` | Reviewer actions | `action` (`approve`/`reject`/`override`/`request_evidence`), `decision_notes` |
| `audit_logs` | Immutable event trail | `action`, `entity_type`, `entity_id`, `details` (JSON), `ip_address` |

### Claim Status Lifecycle

Per the `claims.status` field definition: `Submitted` → `Under Evaluation` → (`Manual Review` →) `Approved` / `Rejected` / `Overridden`.

### Dataset (for model training/benchmarking, distinct from the live application database)

Per `data_dictionary.md`:

| Attribute | Value |
|---|---|
| Total records | 1,500 |
| Columns | 26 (25 predictor/metadata + 1 target label) |
| Class balance | Exactly 500 records per class: Valid Claim, Invalid Claim, Manual Review |
| Train / Val / Test split | 1,050 / 225 / 225 records (70% / 15% / 15%, stratified by class and product category) |
| Categories covered | Laptops (313), Smartphones (307), Washing Machines (301), Audio/Soundbars (298), Smart TVs (281) |
| Noise injection | ~10% of records carry controlled, realistic clerical/edge-case noise |

---

## 15. Machine Learning & Decision Engine

### Tabular Model

- **Algorithm:** `RandomForestClassifier` (scikit-learn), tuned via 5-fold stratified cross-validation.
- **Documented hyperparameters:** `n_estimators=200`, `max_depth=12`, `min_samples_split=4`, `min_samples_leaf=2`, `max_features='sqrt'`, `class_weight='balanced'`, `random_state=42`.
- **Input:** 19 engineered features (temporal boundaries, document-presence booleans, serial-match flag, damage-type risk/frequency encodings, retailer frequency encoding, one-hot product category, repair-history counts).
- **Artifact:** `model/claim_classifier.pkl` (via `joblib`); preprocessing transformers in `model/preprocessing.pkl`.

### Visual (GTM) Model

- **Architecture:** MobileNetV2 backbone, fine-tuned via Google Teachable Machine transfer learning.
- **Input:** 224×224 RGB "Claim Summary Card" images, normalized to `[-1.0, 1.0]`.
- **Training data:** 1,500 rendered Claim Summary Cards.
- **Artifact:** `model/gtm_model/keras_model.h5` + `labels.txt`.
- **Documented failure mode:** raises a `RuntimeError` if TensorFlow or the model file is unavailable — "zero synthetic fallback values" (`PROJECT_REPORT.md` §15.2). Note the corresponding troubleshooting entry describing a Windows/Python 3.14 TensorFlow C++ DLL fallback path (see [Troubleshooting](#21-troubleshooting--faq)), which appears to qualify this "no fallback" statement in at least one documented scenario.

### Deterministic Rule Engine (12 Rules)

`EXCLUDED_DAMAGE_CHECK`, `UNAUTHORIZED_REPAIR`, `WARRANTY_ACTIVE`, `PROOF_OF_PURCHASE_PRESENT`, `SERIAL_NUMBER_MATCH` (incl. PTA DIRBS check for phones), `MANDATORY_DOCUMENTS_COMPLETE`, `FAULT_COVERED`, `REPORTING_PERIOD_VALID`, `LEMON_LAW_CHECK` (≥3 prior major repairs), `PRIOR_REPLACEMENT_CHECK`, `DUPLICATE_CLAIM_CHECK`, `CONTRADICTION_DETECTION`.

### Decision Arbitration Logic

1. Compute the absolute confidence gap between the tabular and visual models' top class.
2. If the two models disagree on class → **Model Disagreement** → escalate to Manual Review.
3. If they agree, classify the gap: **≤15% = Strong Match**, **15–30% = Acceptable Match**, **30–45% = Weak Match**, **>45% = Uncertain Result** (Weak/Uncertain also escalate to Manual Review).
4. Any hard rule failure (e.g., expired warranty, excluded damage) **vetoes** ML consensus regardless of confidence — "Deterministic Override Supremacy" is stated as a hard system constraint.
5. If rules pass and the models are in Strong/Acceptable agreement with top confidence ≥70%, the decision auto-resolves to `Likely Valid` / `Likely Invalid`; otherwise, it escalates to `Manual Review Required`.
6. A natural-language explanation is generated, structured into `factors_supporting`, `factors_opposing`, `rules_passed`, `rules_failed`, and `evidence_needed`.

### Note on Benchmark Figures

The following performance figures are reported consistently across `README.md`, `PROJECT_REPORT.md`, and `TECHNICAL_BLOG_POST.md`:

- **Tabular model test accuracy:** 92.0% (207/225), Macro-F1 0.92, on a held-out 225-record test set.
- **Confusion matrix:** Zero Valid claims misclassified as Invalid and zero Invalid claims misclassified as Valid; all model errors fall into/out of the Manual Review class.
- **Full pipeline (36 unseen claims):** 88.9% model agreement (32/36), 100.0% adjudication accuracy (36/36 correctly classified or safely escalated), with a documented breakdown of 58.3% Strong / 25.0% Acceptable / 5.6% Weak matches and 11.1% flagged disagreements (all successfully escalated).

**These figures are the project's own self-reported results**, as documented in its engineering report. They were **not independently reproduced or verified** as part of producing this README, since no model artifacts, evaluation scripts, or raw prediction logs were available to re-run. Treat them as documented claims rather than externally validated benchmarks, and reproduce them yourself via `python run_model_comparison_pipeline.py` against a running instance if independent verification is required.

---

## 16. Testing

### Framework

**Pytest**, per the `tests/` directory structure documented in `PROJECT_REPORT.md` §17 and `AI_USAGE.md` Entry 6. No pytest-specific package (e.g., `pytest`, `pytest-asyncio`) appears in the provided `requirements.txt`, so a working test run may require installing pytest separately — **this could not be verified** from the files provided.

### How to Run Tests

```bash
# Run the complete test suite
python -m pytest

# Verbose output
python -m pytest -v

# Run specific suites
python -m pytest tests/test_rule_engine.py tests/test_decision_engine.py

# Standalone end-to-end verification scripts
python test_login_end_to_end.py
python verify_all_roles_rbac.py
```

### Documented Test Suites

| Suite | Coverage |
|---|---|
| `tests/conftest.py` | Shared fixtures: `TestClient`, in-memory SQLite DB, seeded entities |
| `tests/test_auth.py` | Authentication, password hashing, token validation |
| `tests/test_claims.py` | Claim submission, status transitions, PDF export |
| `tests/test_decision_engine.py` | Arbitration matrix, consistency tiers, explanations |
| `tests/test_rule_engine.py` | All 12 policy rules, PTA/NTN compliance, lemon-law logic |
| `tests/test_security_rbac.py` | Role privilege enforcement, SQL-injection/XSS/bypass attempts |
| `tests/test_ocr_and_duplicates.py` | OCR field parsing, SHA-256 duplicate detection |
| `tests/test_model_inference.py` | Tabular and GTM model inference guarantees |

### Documented Test Scenario Fixtures (`sample_claims/`)

Eleven fixture claims are documented, each targeting a specific business rule or pipeline behavior: a clean Valid claim, a clean Invalid claim (liquid damage), a borderline Manual Review claim, an expired-warranty claim, a missing-receipt claim, a duplicate (SHA-256 match) claim, a chronologically contradictory claim, a serial-mismatch claim, an unauthorized-repair claim, a boundary-date (last day of warranty) claim, and a model-disagreement claim.

### Test Types Present

Per the suite names above, the documented testing strategy spans **unit tests** (rule engine, decision engine in isolation), **integration tests** (claims lifecycle, auth flow via `TestClient`), **security tests** (RBAC, injection/XSS attempts), and **standalone end-to-end scripts** (login flow, cross-role RBAC) — but not a dedicated browser-based UI/E2E framework (e.g., Playwright/Selenium); frontend verification is described as manual (per `AI_USAGE.md` Entry 5: manual cross-browser and viewport-resizing checks).

---

## 17. Security

The following mechanisms are described in the project documentation (primarily `PROJECT_REPORT.md` §18 and `AI_USAGE.md` Entry 7):

| Mechanism | Description |
|---|---|
| Password hashing | "Salted bcrypt" — see the [dependency discrepancy note](#discrepancy-mlocr-dependencies-vs-requirementstxt) regarding the absence of a bcrypt package in `requirements.txt` |
| Token authentication | JWT, HMAC-SHA256, 24-hour default expiry, `Authorization: Bearer` header transport |
| Role-based authorization | `require_role()` dependency-injection gatekeeper; independently verified via `verify_all_roles_rbac.py` |
| No authentication bypass | Documented as an explicit audit finding — all demo-login/1-click bypass code was searched for and removed |
| SQL injection protection | All database access via parameterized SQLAlchemy ORM queries (no raw SQL string concatenation documented) |
| XSS protection | Frontend text rendering described as sanitizing dynamic HTML content |
| Duplicate/fraud detection | SHA-256 hashing of every uploaded document; identical hashes across claims/claimants flagged |
| Regulatory compliance checks | PTA DIRBS handset verification (smartphone IMEIs) and FBR NTN/STRN sales-tax invoice format validation, enforced as rule-engine rules rather than live third-party API calls |
| Immutable audit logging | Every claim submission, prediction run, status change, and reviewer action logged with actor, timestamp, IP address, and JSON detail payload |
| Environment-based secrets | `JWT_SECRET_KEY` and database connection sourced from environment variables / `.env`, not hardcoded in application logic (though the documentation's own example `.env` value is illustrative only and should never be reused — see Section 6) |

**Security items explicitly stated as tested** (per `test_security_rbac.py` and `verify_all_roles_rbac.py`): SQL-injection attempts, XSS payload handling, and cross-role privilege-bypass attempts.

**Not documented / could not be verified:** rate limiting, CSRF protection specifics, secure file-upload type/size validation rules, dependency vulnerability scanning, and secrets-management practices beyond `.env` usage.

---

## 18. Error Handling & Validation

- **Request validation:** Pydantic models (via FastAPI) are the declared basis for request-body validation across all routers; the standard FastAPI behavior of returning `422 Unprocessable Entity` on schema violations is the expected baseline given this stack, though specific custom validators were not directly inspected.
- **Business-rule validation:** Enforced primarily through the deterministic rule engine at claim-adjudication time (e.g., `WARRANTY_ACTIVE`, `PROOF_OF_PURCHASE_PRESENT`, `CONTRADICTION_DETECTION`) rather than at simple field-level form validation — meaning a structurally valid claim can still fail business rules and be routed to Manual Review or rejection.
- **OCR confidence flags:** Extracted receipt fields carry per-field confidence indicators surfaced to the user in Step 4 of the claim wizard, allowing manual correction before submission (though the exact UI validation behavior on low-confidence fields was not directly inspected).
- **Model inference guards:** The GTM visual classifier is documented to raise a `RuntimeError` (rather than silently degrading) if its model files or TensorFlow runtime are unavailable, so that no claim is silently scored with placeholder/fabricated confidence values — with the caveat noted in Section 15 about a documented platform-specific fallback path.
- **Auth/session errors:** A specific historical bug ("Session Expired" errors on valid seeded credentials, caused by a UTC/local clock-drift mismatch) and its fix (60-second JWT decode leeway) are documented in `AI_USAGE.md`, Entry 8 — indicating this class of error was previously encountered and addressed, though it is worth being aware of if similar symptoms recur in a new environment.

---

## 19. Reports, Export & Documents

| Capability | Description |
|---|---|
| **PDF Claim Certificates** | Generated with **ReportLab** (native document flowables, not a headless-browser HTML-to-PDF converter — the documentation notes this was a deliberate fix to avoid Windows crashes). Includes a branded header with tracking barcode, claimant/product/serial metadata, warranty coverage timeline, the 12-rule audit checklist, and the full multimodal decision breakdown. Downloaded via the claim tracker / Reviewer / Admin views, or directly via `GET /api/claims/{claim_id}/report`. |
| **CSV Export** | Filtered claim-data export from the Admin Dashboard, containing claim attributes, model predictions/confidences, rule-audit results, and reviewer notes, intended for external actuarial analysis. |
| **Batch Comparison Reports** | `run_model_comparison_pipeline.py` produces `reports/unseen_test_claims_report.csv` and the equivalent `.md` file, each a full comparison matrix (actual class, both models' predictions/confidences, agreement flag, confidence gap, consistency tier, rule result, final decision, correctness, and a human-readable explanation) across a configurable number of unseen claims. |
| **Invoices/Receipts** | The system consumes claimant-uploaded receipts/invoices as input (for OCR extraction) rather than generating them; no invoice-generation feature is documented. |
| **Print functionality** | Not explicitly documented beyond standard browser print-to-PDF of the web UI; no dedicated print stylesheet or print API was described. |

---

## 20. Demo / Default Credentials

The following credentials are documented explicitly in the project's own README (§8) as the seeded evaluation accounts. **These are reproduced exactly as provided in the project documentation** — no credentials have been invented, and none beyond these were found in the reviewed files.

| Role | Username | Email Address | Password | Intended Use / Access Scope |
|---|---|---|---|---|
| System Administrator | `admin` | `admin@assurex.com` | `Admin@12345` | System telemetry, model-agreement KPIs, CSV export, audit logs |
| Claim Reviewer | `reviewer` | `reviewer@assurex.com` | `Reviewer@12345` | Manual review queue, evidence inspection, approve/reject/override |
| Service Center | `service_center` | `service@assurex.com` | `Service@12345` | Physical diagnostic intake, repair logging, serial audits |
| Customer | `customer` | `customer@assurex.com` | `Customer@12345` | Product registration, claim filing wizard, OCR verification, status tracking |

Users may authenticate with either their **username** or **email address**, per both `README.md` and `AI_USAGE.md`. A separate `CREDENTIALS.md` file is referenced in the repository structure as containing the same accounts alongside claim-scenario guidance, but its content was not among the files provided for this README.

> ⚠️ These are documented **local evaluation/demo credentials only**. Rotate or disable them before any non-local or production deployment.

---

## 21. Troubleshooting & FAQ

The following are reproduced from the project's own documented FAQ, supplemented with reasonable general guidance where the project's stack makes the cause self-evident.

**Q: "Port 8000 already in use" on startup?**
A: Start on a different port: `python run_server.py --port 8080`, then visit `http://localhost:8080/`.

**Q: TensorFlow C++ DLL warning on Windows with Python 3.14?**
A: Per the documentation, `run_model_comparison_pipeline.py` and `gtm_classifier.py` are designed to automatically detect native C++ runtime ABI incompatibilities and fall back to a "high-fidelity visual telemetry inspector" if native Windows TensorFlow DLLs fail to initialize. (Note: this documented fallback appears to sit alongside the separate statement in Section 15 that the GTM classifier has "zero synthetic fallback" — if you rely on this behavior, verify directly against the actual `gtm_classifier.py` source, which was not available for this README.)

**Q: How do I reset the database to a clean state?**
A: Delete `database/assurex.db` and re-run `python seed_db.py`.

**Q: `pip install -r requirements.txt` succeeds, but the visual classifier or OCR pipeline fails to import?**
A: As noted in Section 4, `requirements.txt` does not include `tensorflow`/`keras` (needed for the GTM visual classifier per the documentation) or `easyocr` (needed for the receipt OCR pipeline per the documentation). Install these separately if you need those specific features, or confirm with the project's actual maintainers whether an alternate, lighter-weight code path is used instead.

**Q: Login fails with a "Session Expired" error immediately after a fresh seed?**
A: This is a previously-documented and fixed issue (`AI_USAGE.md`, Entry 8) caused by UTC/local clock drift interacting with JWT expiry checks. If it recurs, verify your system clock is correctly synchronized and confirm the JWT decode leeway is present in your copy of `backend/auth.py`.

**Q: Database connection errors on startup?**
A: Confirm the `database/` directory is writable and that `DATABASE_URL` in `.env` (if present) points to a valid, accessible path/connection string. The default SQLite configuration should require no additional setup.

**Q: Missing model files (`claim_classifier.pkl`, `keras_model.h5`)?**
A: The documentation states these are pre-bundled in the repository at the exact paths shown in Section 5, Step 4. If they are missing, the tabular/visual classifiers cannot run; re-obtain them from the source repository.

---

## 22. Deployment Guide

The provided documentation is oriented entirely around **local development/evaluation** (virtual environment, SQLite, `python run_server.py`, `--reload` Uvicorn). No dedicated production deployment guide, containerization files (`Dockerfile`, `docker-compose.yml`), process-manager configuration (systemd, Gunicorn worker config), or cloud-specific deployment instructions were found among the provided files.

If deploying AssureX beyond local evaluation, based only on what the documentation does establish, plan to independently address:

- **Production build:** No separate "build" step is documented for the backend (Python is interpreted); the frontend is static HTML/CSS/JS with no documented bundler/build pipeline.
- **Environment configuration:** Set a strong, unique `JWT_SECRET_KEY`; set `ENVIRONMENT=production`; review `LOG_LEVEL`.
- **Backend deployment:** Run Uvicorn without `--reload`, behind a production-grade process manager and reverse proxy (not documented in the provided files — a standard FastAPI/Uvicorn deployment pattern, but not something this project's own documentation specifies).
- **Frontend deployment:** The frontend is served as static assets from the same FastAPI application (per `backend/main.py`'s "static mounting" role in Section 13); no separate frontend deployment target is documented.
- **Database deployment:** Documentation states SQLite by default with PostgreSQL "compatibility" via `DATABASE_URL`, but no PostgreSQL driver is present in `requirements.txt` (see Section 4) — provisioning a real PostgreSQL-backed deployment would require adding that dependency and validating connectivity yourself.
- **TLS/HTTPS, secrets management, container orchestration, horizontal scaling:** Not documented in any of the provided files.

**This section is intentionally conservative** — rather than prescribing a deployment architecture the project's own documentation does not describe, it flags exactly what would need to be defined before a production deployment.

---

## 23. Performance & Architecture

### Architecture (Five-Tier, per `TECHNICAL_BLOG_POST.md` §3 and `PROJECT_REPORT.md` §9–12)

1. **Presentation Tier:** Static HTML/CSS/Vanilla JS SPA served by the FastAPI backend, using Chart.js for visualization.
2. **API/Application Tier:** FastAPI routers (`auth`, `products`, `warranties`, `claims`, `reviews`, `admin`, `export`) with JWT/RBAC middleware.
3. **Intelligence Tier:** The tabular ML model, the visual GTM model, and the deterministic rule engine, run — per the documented sequence diagram — **in parallel** for a given claim.
4. **Arbitration Tier:** The decision engine, which reconciles the three parallel outputs into one final, explainable decision.
5. **Persistence Tier:** SQLAlchemy ORM over SQLite (default) / PostgreSQL ("compatible," per documentation), with 9 normalized entities plus JSON policy/threshold configuration files.

### Documented Performance Targets

- **End-to-end claim adjudication:** ≤500 ms per claim (tabular + visual + rule + arbitration combined), per NFR-01 in `PROJECT_REPORT.md` §8.2.
- **Page load time:** ≤1.5 seconds under standard network conditions (same source).
- **Model accuracy target:** ≥88% test accuracy and ≥0.85 Macro-F1 (actual reported result: 92.0% / 0.92 — see [Note on Benchmark Figures](#note-on-benchmark-figures)).
- **Determinism:** Identical claim inputs are required to produce identical rule-engine and decision-engine outputs across runs (NFR-03).

### Key Architectural Decision: Deterministic Override Supremacy

A documented, explicit system constraint: machine-learning confidence — no matter how high — can never override a hard rule-engine failure (e.g., an expired warranty or an excluded-damage finding). This is presented as the core design principle distinguishing AssureX from either a pure-ML or a pure-rules approach (`TECHNICAL_BLOG_POST.md` §2).

---

## 24. System Limitations

As explicitly documented in `PROJECT_REPORT.md` §19:

1. **OCR quality dependency** — severely faded thermal receipts or low-resolution photographs can degrade OCR extraction accuracy, requiring human verification.
2. **Vision model scope** — the GTM model evaluates standardized Claim Summary Cards; it does not perform raw pixel-level semantic segmentation on arbitrary physical hardware photos (e.g., distinguishing a micro-crack from a screen-protector scratch).
3. **Cross-platform TensorFlow C++ ABI issues** — on bleeding-edge Python releases (e.g., Python 3.14 on Windows), precompiled native TensorFlow wheels may require the documented visual-inspection fallback.
4. **Duplicate-detection scope** — SHA-256 duplicate detection relies on the database history of the current deployment instance only; it cannot detect duplicates against data outside that instance.

### Additional Limitations Identified During Documentation Review (Not Explicitly Stated by the Project)

- **Dependency-manifest gaps** — as detailed in Section 4, `tensorflow`, `easyocr`, and a bcrypt-compatible hashing library are described as core to the system but are absent from `requirements.txt`, which could block a from-scratch setup relying solely on that file.
- **No PostgreSQL driver** — "PostgreSQL compatible" is stated but not backed by a corresponding driver package in `requirements.txt`.
- **No pytest dependency listed** — the documented test suites require `pytest`, which is not present in `requirements.txt`.
- **No production deployment artifacts** — no Dockerfile, process-manager config, or infrastructure-as-code was found among the reviewed files (see Section 22).

---

## 25. Future Improvements

Presented here strictly as **documented, potential future enhancements** — none of the following are implemented in the current system, per `PROJECT_REPORT.md` §20:

1. **Deep visual defect segmentation** — integrate an active object-detection/segmentation model (e.g., YOLOv10 or Mask R-CNN) trained on microscopic hardware-inspection imagery to detect solder fractures, blown capacitors, and liquid residue directly from physical device photos.
2. **Direct PTA DIRBS API gateway** — replace the simulated regulatory-compliance rule checks with a live integration against telecom regulatory web services for real-time IMEI status verification.
3. **Consumer mobile application** — native iOS/Android apps with on-device camera guidance for barcode scanning and receipt capture.
4. **Automated LLM policy synthesizer** — a Retrieval-Augmented Generation (RAG) assistant (referenced as potentially using Gemini) to convert natural-language warranty contract PDFs into structured rule-engine configuration automatically.
5. **Decentralized warranty registry** — blockchain-backed digital warranty tokens (NFTs / verifiable credentials) issued at point-of-sale, aimed at eliminating receipt fraud entirely.

---

## 26. License

The project's own README badge states **MIT License**. **No `LICENSE` file was included among the files provided for this README**, so the exact license text and any additional terms could not be independently verified. Treat the badge as the project's stated intent and confirm against an actual `LICENSE` file in the repository before relying on it for legal purposes.

---

## 27. Developer / Credits

- **Project / Document Author:** "AssureX Core Engineering Team" (as credited in `PROJECT_REPORT.md`'s document control header and closing line).
- **AI-Assisted Engineering:** Extensive AI-assisted development is disclosed and logged in detail in `AI_USAGE.md`, which credits **"Antigravity AI (Google DeepMind)"** as the AI tool used across all eleven logged engineering interactions (GTM classifier, decision engine, backend/API architecture, OCR module, frontend SPA, test suites, security audit, auth bug fix, PDF generation, batch evaluation pipeline, and the project report itself), with each entry documenting the human engineer's specific changes, oversight, and testing performed.
- **No individual developer name, contact information, or organizational affiliation** (beyond the team label above) was found in any of the files provided for this README.

---

*This README was compiled from the AssureX Claim Engine's own project documentation (`README.md`, `PROJECT_REPORT.md`, `TECHNICAL_BLOG_POST.md`, `DEMO_VIDEO_SCRIPT.md`, `data_dictionary.md`, `AI_USAGE.md`, `requirements.txt`). Statements are attributed to their source documents wherever practical, and points that could not be cross-verified against the concrete `requirements.txt` dependency manifest are flagged explicitly rather than presented as confirmed fact.*
