"""
main.py
=======
Gemini Tool-Calling RAG API — optimised for token efficiency.

Two request tools available to the Gemini Agent:
  1. Seismic / retrofit tool -> deterministic engineering parameters
  2. Database Search tool    -> Chroma vector store (Strictly Zero Hallucination)

Engineering tools are based on peer-reviewed BUET/CDMP literature.
"""

from __future__ import annotations

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent





# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — ENGINEERING CONSTANTS & LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

SoilType = Literal["soft_filled", "medium", "hard"]
InterventionType = Literal[
    "column_jacketing", "shear_wall", "steel_moment_frame",
    "frp_wrapping", "full_soft_story",
]
RiskTier = Literal["Critical", "High", "Moderate", "Low"]

SOIL_AMPLIFICATION: dict = {
    "soft_filled": 5.5,
    "medium":      2.0,
    "hard":        1.1,
}

LIQUEFACTION_RISK: dict = {
    "soft_filled": True,
    "medium":      True,
    "hard":        False,
}

YEAR_SCORES = [
    (0,    1993, 40),
    (1993, 2007, 30),
    (2007, 2021, 15),
    (2021, 9999,  0),
]

SOFT_STORY_PENALTY = 30
SOIL_SCORE_MAP: dict = {"soft_filled": 30, "medium": 18, "hard": 6}

RETROFIT_LOOKUP: dict = {
    "column_jacketing":   (550,   1_650),
    "shear_wall":         (1_500, 4_700),
    "steel_moment_frame": (2_000, 6_000),
    "frp_wrapping":       (400,   1_250),
    "full_soft_story":    (2_500, 5_500),
}

# Short summaries used in /ask agent responses (token-efficient)
RETROFIT_SCOPE_SHORT: dict = {
    "column_jacketing":   "RC jacketing of ground-floor columns with additional bars and self-compacting concrete jacket (min 75 mm).",
    "shear_wall":         "New RC shear wall in open bays anchored to existing footing, designed per BNBC 2020.",
    "steel_moment_frame": "Steel moment frame in open parking bays with chemical-anchor base plates; maintains parking access.",
    "frp_wrapping":       "CFRP/GFRP column wrapping for ductility/confinement. Combine with shear walls for full soft-story fix.",
    "full_soft_story":    "Full package: column jacketing + RC shear walls (≥2 bays/direction) + FRP wrapping + joint strengthening.",
}

# Full scope used only on the /retrofit direct endpoint
RETROFIT_SCOPE_FULL: dict = {
    "column_jacketing": (
        "RC Jacketing of existing ground-floor columns.\n"
        "  • Chip cover, clean and roughen surface, apply bonding agent.\n"
        "  • Place additional longitudinal bars + closely-spaced ties.\n"
        "  • Pour self-compacting concrete jacket (min 75 mm thick).\n"
        "  Typical per-column cost: BDT 80,000–200,000."
    ),
    "shear_wall": (
        "New RC Shear Wall addition at ground-floor open bays.\n"
        "  • Design for seismic base shear per BNBC 2020.\n"
        "  • Core-drill connection to existing footing.\n"
        "  • Full-height RC wall with boundary elements + slab anchors.\n"
        "  Typical per-bay cost: BDT 300,000–700,000."
    ),
    "steel_moment_frame": (
        "Steel Moment Frame insertion in open parking bays.\n"
        "  • Fabricate and galvanise steel columns and moment connections.\n"
        "  • Anchor base plates with chemical anchors to existing slab.\n"
        "  • Weld/bolt rigid moment connections at top; fire-protect all steel.\n"
        "  • Maintains parking access with minimal visual obstruction.\n"
        "  Typical per-bay cost: BDT 400,000–900,000."
    ),
    "frp_wrapping": (
        "Fibre-Reinforced Polymer (FRP) Column Wrapping.\n"
        "  • Prepare column surface; apply epoxy primer.\n"
        "  • Wrap with CFRP/GFRP sheets in circumferential direction.\n"
        "  • Protective epoxy top coat.\n"
        "  NOTE: FRP provides ductility/confinement — NOT lateral strength.\n"
        "  Combine with shear walls for full soft-story correction.\n"
        "  Typical per-column cost: BDT 60,000–150,000."
    ),
    "full_soft_story": (
        "Full Soft-Story Retrofit Package (recommended for Critical-tier buildings).\n"
        "  1. RC Column Jacketing of ALL ground-floor columns.\n"
        "  2. New RC Shear Walls in ≥2 bays per direction.\n"
        "  3. FRP wrapping on remaining columns for added ductility.\n"
        "  4. Connection strengthening at ground-floor slab/beam joints.\n"
        "  5. Structural engineer inspections at each stage.\n"
        "  6. Engineering/drawing fees: BDT 1–1.5 lakh (not included above).\n"
        "  Total: 8–15 % of building replacement value (CDMP/HBR).\n"
        "  Expect 4–8 weeks of disruption on ground floor."
    ),
}

_EXCHANGE_RATE = 110.0

MAX_HISTORY = 6  # keep last 3 exchanges (6 messages) for multi-turn calls


def _year_score(year: int) -> int:
    for y_from, y_to, score in YEAR_SCORES:
        if y_from <= year < y_to:
            return score
    return 0


def calculate_vulnerability_score(soil_type: str, construction_year: int, soft_story: bool) -> dict:
    if soil_type not in SOIL_AMPLIFICATION:
        return {"error": f"soil_type must be one of {list(SOIL_AMPLIFICATION)}."}

    soil_score = SOIL_SCORE_MAP[soil_type]
    year_score = _year_score(construction_year)
    ss_score   = SOFT_STORY_PENALTY if soft_story else 0
    total      = min(soil_score + year_score + ss_score, 100)

    if total >= 70:   tier = "Critical"
    elif total >= 50: tier = "High"
    elif total >= 30: tier = "Moderate"
    else:             tier = "Low"

    amp = SOIL_AMPLIFICATION[soil_type]
    liq = LIQUEFACTION_RISK[soil_type]

    notes = []
    if soil_type == "soft_filled":
        notes.append(
            f"Soft/filled Holocene soil amplifies ground shaking ~{amp}× "
            "(range 3.1–8.5× per Ansary & Rahman 2013; Khair et al. 2024)."
        )
    if construction_year < 1993:
        notes.append("Building predates BNBC 1993 — zero seismic detailing was required at construction.")
    elif construction_year < 2007:
        notes.append("Built to BNBC 1993 (Z=0.15g) — deficient relative to BNBC 2020 (+27.5% base shear).")
    elif construction_year < 2021:
        notes.append("Built to BNBC 2006 — moderate code compliance, below BNBC 2020 standard.")

    if soft_story:
        notes.append(
            "Open ground-floor (piloti/parking) is the #1 collapse mode in Dhaka "
            "(Ahmed & Morita 2018, 40.32% OOB importance)."
        )
    if liq:
        notes.append(
            "Site soil susceptible to earthquake-induced liquefaction "
            "(Ansary et al. 2024 — >60% of DMDP area at moderate–high risk)."
        )

    return {
        "risk_tier": tier,
        "total_score": total,
        "sub_scores": {"soil": soil_score, "code_year": year_score, "soft_story": ss_score},
        "site_amplification_factor": amp,
        "liquefaction_susceptible": liq,
        "notes": notes,
        "citations": [
            "[1] Ansary & Rahman (2013). Environ. Earth Sci. 70(2). doi:10.1007/s12665-012-2141-x",
            "[2] Ansary et al. (2024). Natural Hazards 120(11). doi:10.1007/s11069-024-06586-1",
            "[3] Khair, Hore & Ansary (2024). Indian Geotech. J. doi:10.1007/s40098-024-00931-8",
            "[4] Ahmed & Morita (2018). Sustainability 10(4), 1106. doi:10.3390/su10041106",
            "[7] BNBC (2020). Bangladesh National Building Code.",
        ],
    }


def estimate_retrofit_cost(intervention_type: str, approximate_sqft: int, full_scope: bool = False) -> dict:
    if intervention_type not in RETROFIT_LOOKUP:
        return {"error": f"intervention_type must be one of {list(RETROFIT_LOOKUP)}."}

    rate_low, rate_high = RETROFIT_LOOKUP[intervention_type]
    bdt_low  = rate_low  * approximate_sqft
    bdt_high = rate_high * approximate_sqft
    usd_low  = round(bdt_low  / _EXCHANGE_RATE)
    usd_high = round(bdt_high / _EXCHANGE_RATE)

    scope = RETROFIT_SCOPE_FULL[intervention_type] if full_scope else RETROFIT_SCOPE_SHORT[intervention_type]

    return {
        "intervention_type": intervention_type,
        "ground_floor_sqft": approximate_sqft,
        "cost_range_bdt": {"low": bdt_low, "high": bdt_high},
        "cost_range_usd": {"low": usd_low, "high": usd_high},
        "exchange_rate_note": f"1 USD ≈ {_EXCHANGE_RATE} BDT (2025 estimate)",
        "scope_summary": scope,
        "methodology_note": (
            "Pre-design estimates only. Must be verified by a licensed structural "
            "engineer with as-built drawings. Rates assume Dhaka 2024–25 market."
        ),
        "citations": [
            "[5] CDMP (2009). Risk Assessment of Dhaka City Corporation Areas.",
            "[6] HBR/CDMP. Retrofitting: Bangladesh Perspective. hbr.center",
            "[7] BNBC (2020). Bangladesh National Building Code 2020.",
        ],
    }


def _pick_intervention(tier: str, soft_story: bool) -> str:
    if tier == "Critical":            return "full_soft_story"
    if tier == "High" and soft_story: return "shear_wall"
    if tier == "High":                return "column_jacketing"
    if tier == "Moderate":            return "frp_wrapping"
    return "frp_wrapping"


def trim_history(messages: list) -> list:
    """Keep only the last MAX_HISTORY messages to limit context growth."""
    return messages[-MAX_HISTORY:] if len(messages) > MAX_HISTORY else messages


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — FORMATTING (used only by direct /vulnerability and /retrofit endpoints)
# ═══════════════════════════════════════════════════════════════════════════════

def _format_vulnerability(v: dict) -> str:
    if "error" in v:
        return f"⚠ Vulnerability tool error: {v['error']}"
    lines = [
        "━" * 58,
        "  🏗  SEISMIC VULNERABILITY ASSESSMENT",
        "━" * 58,
        f"  Risk Tier              : {v['risk_tier']}",
        f"  Composite Score        : {v['total_score']} / 100",
        f"    Soil sub-score       : {v['sub_scores']['soil']} / 30",
        f"    Code-year sub-score  : {v['sub_scores']['code_year']} / 40",
        f"    Soft-story penalty   : {v['sub_scores']['soft_story']} / 30",
        f"  Site Amplification     : {v['site_amplification_factor']}×",
        f"  Liquefaction Risk      : {'YES ⚠' if v['liquefaction_susceptible'] else 'No'}",
        "",
        "  Assessment Notes:",
    ]
    for note in v["notes"]:
        lines.append(f"    • {note}")
    lines += ["", "  Citations:"]
    for c in v["citations"]:
        lines.append(f"    {c}")
    lines.append("━" * 58)
    return "\n".join(lines)


def _format_retrofit(r: dict) -> str:
    if "error" in r:
        return f"⚠ Retrofit tool error: {r['error']}"
    bdt = r["cost_range_bdt"]
    usd = r["cost_range_usd"]
    lines = [
        "━" * 58,
        "  🔧  RETROFIT COST ESTIMATE",
        "━" * 58,
        f"  Intervention           : {r['intervention_type'].replace('_', ' ').title()}",
        f"  Ground Floor Area      : {r['ground_floor_sqft']:,} sq ft",
        f"  Cost Range (BDT)       : ৳{bdt['low']:,} – ৳{bdt['high']:,}",
        f"  Cost Range (USD ~)     : ${usd['low']:,} – ${usd['high']:,}",
        f"  {r['exchange_rate_note']}",
        "",
        "  Scope of Work:",
        f"    {r['scope_summary']}",
        "",
        f"  ⚠ {r['methodology_note']}",
        "",
        "  Citations:",
    ]
    for c in r["citations"]:
        lines.append(f"    {c}")
    lines.append("━" * 58)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — LANGCHAIN TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def assess_seismic_risk_and_retrofit_tool(
    soil_type: str = Field(description="Must be 'soft_filled', 'medium', or 'hard'. Map Mirpur, Bashundhara, Badda, Old Dhaka to 'soft_filled'. Map Dhanmondi, Gulshan, Uttara to 'medium'. Map Cantonment to 'hard'. Default is 'medium'."),
    construction_year: int = Field(description="The 4-digit construction year. Default is 1995."),
    soft_story: bool = Field(description="True if the building has an open ground floor, parking, piloti, or soft story. Default is False."),
    ground_floor_sqft: int = Field(description="Ground floor area in square feet. Default is 2000."),
    intervention_type: str = Field(description="Specific retrofit requested. Must be 'column_jacketing', 'shear_wall', 'steel_moment_frame', 'frp_wrapping', 'full_soft_story', or 'auto'. Default is 'auto'.")
) -> dict:
    """
    Evaluates deterministic seismic risk and retrofit cost for Dhaka buildings.
    Call this tool ONLY if the user asks about seismic vulnerability, building earthquake risk,
    or retrofitting costs/methods for a specific property.
    Returns structured JSON — the frontend is responsible for formatting.
    """
    vuln = calculate_vulnerability_score(soil_type, construction_year, soft_story)
    if "error" in vuln:
        return {"error": vuln["error"]}

    if intervention_type == "auto" or intervention_type not in RETROFIT_LOOKUP:
        intervention_type = _pick_intervention(vuln.get("risk_tier", "High"), soft_story)

    retrofit = estimate_retrofit_cost(intervention_type, ground_floor_sqft, full_scope=False)
    if "error" in retrofit:
        return {"error": retrofit["error"]}

    return {
        "parameters": {
            "soil_type": soil_type,
            "construction_year": construction_year,
            "soft_story": soft_story,
            "ground_floor_sqft": ground_floor_sqft,
            "recommended_intervention": intervention_type,
        },
        "vulnerability": vuln,
        "retrofit": retrofit,
        "disclaimer": (
            "Preliminary screening tool based on BUET/CDMP research. "
            "Does NOT replace a site-specific structural engineering assessment. "
            "Engage a licensed Bangladeshi structural engineer before any retrofitting works."
        ),
    }


@tool
def search_database_tool(
    query: str = Field(description="The search query to look up in the vector database.")
) -> str:
    """
    Searches the local engineering knowledge base for context.
    Use for general questions about engineering, BNBC, guidelines, or any informational question.
    """
    docs = vectorstore.similarity_search(query, k=5)
    if not docs:
        return "NO_DOCUMENTS_FOUND"
    return "\n\n".join(doc.page_content for doc in docs)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D — APP SETUP & AGENT INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

app = FastAPI(title="Gemini Tool-Calling RAG + Seismic Risk API", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")
"""

**Step 3 — Open the app via FastAPI, not file://**

Instead of double-clicking the HTML, go to:

http://127.0.0.1:8000/static/index.html
"""


print("⏳ Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("⏳ Connecting to Chroma vector store...")
vectorstore = Chroma(
    persist_directory="./chroma_db_project",
    embedding_function=embeddings
)

_SYSTEM_PROMPT = """You are a structural engineering assistant for Dhaka. Rules:

1. Seismic/retrofit questions → call assess_seismic_risk_and_retrofit_tool, then present the result
   in clear, readable plain text using this structure:
   - Risk tier and total score
   - Sub-scores (soil, code year, soft story)
   - Site amplification factor and liquefaction risk
   - Assessment notes (as bullet points)
   - Recommended intervention with cost range in both BDT and USD
   - Scope summary
   - Disclaimer
   - Citations
   Never dump raw JSON. Never skip the disclaimer or citations.

2. All other domain questions → search_database_tool.

3. If search_database_tool returns "NO_DOCUMENTS_FOUND" or contains no relevant answer, reply ONLY:
   "This information is not found in the database."

Never use internal training knowledge for domain questions."""

print("⏳ Initializing Gemini LLM Agent...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY,
)

tools = [assess_seismic_risk_and_retrofit_tool, search_database_tool]

agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    prompt=_SYSTEM_PROMPT,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E — ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status":       "✅ Gemini Tool-Calling API is running (v4.0.0 — token-optimised)",
        "version":      "4.0.0",
        "model":        "gemini-2.5-flash",
        "chunks_in_db": vectorstore._collection.count(),
        "endpoints": {
            "ask":           "/ask?query=your+question",
            "vulnerability": "/vulnerability?soil_type=soft_filled&construction_year=1995&soft_story=true",
            "retrofit":      "/retrofit?intervention_type=full_soft_story&sqft=2400",
        },
    }


@app.get("/ask")
async def ask_ai(query: str):
    if not query or not query.strip():
        return {"reply": "Please provide a non-empty query."}
    try:
        messages = trim_history([HumanMessage(content=query)])
        result = agent_executor.invoke({"messages": messages})
        
        # Determine which tool was used by looking at the message history
        # Extract plain text — Gemini returns content as either a string
        # or a list of content blocks [{"type": "text", "text": "..."}]
        raw = result["messages"][-1].content
        if isinstance(raw, list):
            last_msg = "".join(
                block.get("text", "") for block in raw
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            last_msg = raw
        tool_used = "rag" # Default
        
        for msg in result["messages"]:
            if hasattr(msg, "name"):
                if msg.name == "assess_seismic_risk_and_retrofit_tool":
                    tool_used = "seismic_engineering_tools"
                elif msg.name == "search_database_tool":
                    tool_used = "project_calc_tool" # Matches your JS logic

        return {"reply": last_msg, "tool_used": tool_used}
    except Exception as e:
        return {"reply": f"Error: {str(e)}", "tool_used": "rag"}

@app.get("/vulnerability")
async def vulnerability_endpoint(
    soil_type: str,
    construction_year: int,
    soft_story: bool = False,
):
    """Direct deterministic vulnerability calculation — no LLM involved."""
    return calculate_vulnerability_score(soil_type, construction_year, soft_story)


@app.get("/retrofit")
async def retrofit_endpoint(
    intervention_type: str,
    sqft: int,
):
    """Direct retrofit cost estimate with full scope-of-work text — no LLM involved."""
    return estimate_retrofit_cost(intervention_type, sqft, full_scope=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION F — RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)