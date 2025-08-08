# Dream 4 Degree - Streamlit Frontend

import streamlit as st
import json
import pandas as pd
from typing import List, Dict, Any
import sys
import os
from pathlib import Path

# Ensure project root is on sys.path so `import backend...` works when running from frontend/
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

st.set_page_config(
    page_title="Dream 4 Degree",
    page_icon=str(Path(__file__).parent / "assets" / "Grad Cap via AI.svg"),
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css():
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        :root {
            --cuny-blue: #0033A0;
            --cuny-blue-2: #0050FF;
            --bg-soft: #F5F7FB;
            --text: #1F2937;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: var(--bg-soft); border-radius: 12px; padding: 6px; }
        .stTabs [data-baseweb="tab"] { height: 44px; background-color: #fff; border-radius: 8px; color: var(--text); font-weight: 600; border: 1px solid #E5E7EB; }
        .stTabs [aria-selected="true"] { background-color: var(--cuny-blue); color: #fff; border: 1px solid var(--cuny-blue-2); }
        .main-header { display:flex; align-items:center; gap:12px; padding:8px 0 12px; border-bottom:1px solid #E5E7EB; }
        .main-header h1 { margin:0; color: var(--cuny-blue); }
        .main-header p { margin:0; color:#4B5563; }
        .recommendation-card { background: linear-gradient(135deg, var(--cuny-blue) 0%, var(--cuny-blue-2) 100%); color: white; padding: 1.25rem; border-radius: 10px; margin-bottom: 1rem; box-shadow: 0 4px 8px rgba(0,0,0,0.08); }
        .fit-score { background: #10B981; color: white; padding: 0.4rem 0.8rem; border-radius: 16px; font-weight: 700; display: inline-block; }
        .source-pill { display:inline-block; background:#EEF2FF; color: var(--cuny-blue); padding:6px 10px; border-radius:14px; margin: 4px 6px 0 0; font-size: 0.9rem; }
        </style>
        """, unsafe_allow_html=True)

load_css()

# Import backend directly instead of making HTTP calls

# Direct imports from backend (use absolute import so Pylance can resolve)
try:
    from backend.advisor_agent import CUNYAdvisorAgent, UserProfile
    BACKEND_AVAILABLE = True
except ImportError as e:
    st.error(f"Backend import error: {e}")
    BACKEND_AVAILABLE = False

# Initialize global advisor agent
if BACKEND_AVAILABLE:
    @st.cache_resource
    def get_advisor_agent():
        """Initialize and cache the advisor agent"""
        try:
            return CUNYAdvisorAgent()
        except Exception as e:
            st.error(f"Failed to initialize advisor agent: {e}")
            return None
else:
    def get_advisor_agent():
        return None

def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """
    Call backend functions directly instead of making HTTP requests
    """
    print(f"DEBUG: call_api called with endpoint='{endpoint}', method='{method}', data={data}")
    
    if not BACKEND_AVAILABLE:
        print("DEBUG: Backend not available")
        return {"error": "Backend not available"}
    
    advisor_agent = get_advisor_agent()
    print(f"DEBUG: advisor_agent: {advisor_agent}")
    if not advisor_agent:
        print("DEBUG: Advisor agent not initialized")
        return {"error": "Advisor agent not initialized"}
    
    try:
        # Handle different endpoints by calling backend functions directly
        if endpoint == "/api/search" and method == "POST":
            # Search programs using ChromaDB
            query = data.get("query", "")
            n_results = data.get("n_results", 10)
            
            print(f"DEBUG: ChromaDB Search query: '{query}'")
            
            # Use ChromaDB for search - no fallbacks
            results = advisor_agent.chroma_manager.search_programs(query, n_results)
            print(f"DEBUG: ChromaDB Raw results: {results}")
            
            if results and results.get('metadatas') and results.get('metadatas')[0]:
                search_results = []
                for i, metadata in enumerate(results['metadatas'][0]):
                    search_results.append({
                        "program_name": metadata.get('program_name'),
                        "college": metadata.get('college'),
                        "degree_type": metadata.get('degree_type'),
                        "cip_title": metadata.get('cip_title'),
                        "tap_eligible": metadata.get('tap_eligible'),
                        # Additional metadata to support richer comparisons
                        "cip_code": metadata.get('cip_code'),
                        "irp_code": metadata.get('irp_code'),
                        "date_program_established": metadata.get('date_program_established'),
                        "record_type": metadata.get('record_type'),
                        "period": metadata.get('period'),
                        "rank": i + 1
                    })
                print(f"DEBUG: ChromaDB search_results: {search_results}")
                return {
                    "success": True,
                    "query": query,
                    "results": search_results,
                    "total_found": len(search_results)
                }
            else:
                print("DEBUG: ChromaDB returned no results")
                return {
                    "success": True,
                    "query": query,
                    "results": [],
                    "total_found": 0
                }
            
        elif endpoint == "/api/recommendations" and method == "POST":
            # Get recommendations
            user_profile = UserProfile(
                interests=data.get("interests", []),
                preferred_location=data.get("preferred_location"),
                budget_max=data.get("budget_max"),
                career_goals=data.get("career_goals", []),
                values=data.get("values", []),
                academic_strengths=data.get("academic_strengths", [])
            )
            
            recommendations = advisor_agent.get_recommendations(user_profile)
            recommendations_data = []
            for rec in recommendations:
                recommendations_data.append({
                    "program_name": rec.program_name,
                    "college": rec.college,
                    "degree_type": rec.degree_type,
                    "estimated_cost": rec.estimated_cost,
                    "fit_score": rec.fit_score,
                    "reasoning": rec.reasoning,
                    "pros": rec.pros,
                    "cons": rec.cons
                })
            
            return {
                "success": True,
                "recommendations": recommendations_data,
                "total_found": len(recommendations_data)
            }
            
        elif endpoint == "/api/majors":
            # Get popular majors - load from data file
            import json
            data_dir = Path(__file__).parent.parent / "data"
            majors_file = data_dir / "CUNY_Top_15_Majors.json"
            
            if majors_file.exists():
                with open(majors_file, 'r', encoding='utf-8') as f:
                    majors_data = json.load(f)
                return {
                    "success": True,
                    "majors": majors_data.get('Top15Majors', [])
                }
            return {"success": False, "error": "Majors data not found"}
            
        else:
            return {"success": False, "error": f"Endpoint {endpoint} not implemented"}
            
    except Exception as e:
        return {"success": False, "error": f"Backend error: {str(e)}"}

# ---------- RAG helpers (Ask the Advisor) ----------
# History manager is optional; guard import so the app can still run if backend isn't available.
if BACKEND_AVAILABLE:
    try:
        from backend.history_manager import HistoryManager
        _history = HistoryManager()
    except Exception as e:
        st.warning(f"History disabled (import error): {e}")
        class _NullHistory:
            def log_qa(self, *_, **__):
                return None
        _history = _NullHistory()
else:
    class _NullHistory:
        def log_qa(self, *_, **__):
            return None
    _history = _NullHistory()

def _get_user_id() -> str:
    # Try to use a stable cookie if Streamlit supports it; otherwise fallback to a session token
    # Streamlit doesn't expose cookies directly without extras; use a session key fallback.
    if 'user_id' not in st.session_state:
        import uuid
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id
def _build_profile_from_quiz() -> Any:
    """Build a lightweight UserProfile from stored quiz responses if available."""
    try:
        qr = st.session_state.get('quiz_responses', {})
        if not qr:
            return None

        interests: List[str] = []
        # Section 2 (interests & passions)
        s2 = qr.get('section_2', {})
        interests += s2.get('favorite_subjects', [])
        interests += s2.get('passion_areas', [])
        # Section 4 (industry interests)
        s4 = qr.get('section_4', {})
        interests += s4.get('industry_interests', [])

        # Preferred location (first choice if any)
        s6 = qr.get('section_6', {})
        locs = s6.get('location_pref', [])
        preferred_location = locs[0] if isinstance(locs, list) and locs else None

        career_goal = s4.get('career_goal')
        core_values = s4.get('core_values', [])
        s3 = qr.get('section_3', {})
        academic_strengths = s3.get('academic_strengths', [])

        return UserProfile(
            interests=list(dict.fromkeys([i for i in interests if i])),  # de-dup
            preferred_location=preferred_location,
            budget_max=None,  # quiz stores budget categorically; leave None for now
            career_goals=[career_goal] if career_goal else [],
            values=core_values,
            academic_strengths=academic_strengths,
        )
    except Exception:
        return None


def ask_advisor_ui(key_prefix: str, use_quiz_profile: bool = True):
    """Render the Ask the Advisor input + answer.

    key_prefix ensures Streamlit widget keys don't collide across pages.
    If use_quiz_profile, will attach a lightweight profile built from quiz answers.
    """
    st.caption("Tip: Ask focused questions like 'best CS programs in Queens' for better answers 🔎")

    q = st.text_input(
        "Ask a question about CUNY programs",
        key=f"rag_q_{key_prefix}",
        placeholder="e.g., best CS programs in Queens"
    )
    if not q:
        return

    agent = get_advisor_agent()
    if not agent:
        st.error("Advisor not available.")
        return

    with st.spinner("Thinking..."):
        profile = _build_profile_from_quiz() if use_quiz_profile else None
        try:
            result = agent.ask_advisor(q, profile)
        except Exception as e:
            st.error(f"Advisor error: {e}")
            return

    answer = result.get("answer", "")
    st.write(answer)
    sources = result.get("sources", [])
    if sources:
        with st.expander("Sources (top matches)"):
            for s in sources:
                st.markdown(f"- {s['program_name']} | {s['college']} | {s['degree_type']} | {s.get('cip_title','')}")

    # Persist Q&A to history
    try:
        user_id = _get_user_id()
        meta = {"sources": sources[:5]}
        _history.log_qa(user_id, q, answer, meta=meta)
    except Exception:
        pass

    # Debug panel (opt-in)
    st.session_state["rag_last_question"] = q
    st.session_state["rag_last_sources"] = sources
    with st.expander("Debug (context)"):
        st.markdown(f"**Last question:** {st.session_state.get('rag_last_question','')}")
        srcs = st.session_state.get("rag_last_sources") or []
        if srcs:
            st.markdown("**Top matches used:**")
            for s in srcs:
                st.markdown(f"- {s.get('program_name','')} — {s.get('college','')} ({s.get('degree_type','')})")

def display_header():
    """Top header with CUNY-themed brand bar and logo (SVG)."""
    assets_dir = Path(__file__).parent / "assets"
    logo_path = assets_dir / "Grad Cap via AI.svg"
    logo_img_tag = None
    try:
        if logo_path.exists():
            import base64
            svg_bytes = logo_path.read_bytes()
            b64 = base64.b64encode(svg_bytes).decode("ascii")
            # Explicitly constrain size to avoid oversized rendering in some environments
            logo_img_tag = (
                f'<img alt="CUNY logo" src="data:image/svg+xml;base64,{b64}" '
                f'width="40" height="40" style="width:40px;height:40px;object-fit:contain;" />'
            )  # safe as data URI
    except Exception:
        logo_img_tag = None

    if logo_img_tag:
        st.markdown(
            f"""
            <div class="main-header">
              <div class="app-brand">
                <div class="app-logo">{logo_img_tag}</div>
                <div class="app-title">
                  <h1>Dream 4 Degree</h1>
                  <p>AI-Powered Major and Campus Selection Assistant</p>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="main-header">
                <h1>Dream 4 Degree</h1>
                <p>AI-Powered Major and Campus Selection Assistant</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

def display_navigation():
    return st.tabs([
    "🏛️ Home",
    "🧩 Quiz",
    "🤖 Advisor",
    "🔎 Search",
    "🧮 Calculator",
    "📊 Compare",
    ])

def home_page():
    st.header("Welcome to Dream 4 Degree")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            ### Personalized Recommendations
            Get AI-powered suggestions based on your interests, goals, and budget.
            """
        )
    
    with col2:
        st.markdown(
            """
            ### Data-Driven Insights
            Our recommendations are based on real CUNY program data and cost analysis.
            """
        )
    
    with col3:
        st.markdown(
            """
            ### Easy Comparison
            Compare programs side-by-side to make informed decisions.
            """
        )
    
    st.markdown("---")
    
    st.subheader("Quick Program Search")
    st.markdown('<div class="app-card-start"></div>', unsafe_allow_html=True)
    with st.container():
        search_query = st.text_input("Search CUNY programs (e.g., 'psychology brooklyn', 'computer science')")
        st.caption("Tip: Try keywords like 'computer science queens' or 'nursing brooklyn' 🔎")

        # Initialize session state for search results
        if 'home_search_results' not in st.session_state:
            st.session_state.home_search_results = None

        if st.button("Search") and search_query:
            print(f"DEBUG: Search button clicked with query: '{search_query}'")
            with st.spinner("Searching programs..."):
                search_data = {'query': search_query, 'n_results': 10}
                print(f"DEBUG: About to call API with search_data: {search_data}")
                result = call_api("/api/search", "POST", search_data)
                print(f"DEBUG: API call result: {result}")

                if result.get('success'):
                    st.session_state.home_search_results = result.get('results', [])
                    print(f"DEBUG: Stored {len(result.get('results', []))} results in session state")
                else:
                    st.session_state.home_search_results = []
                    error_msg = result.get('error', 'Unknown error occurred')
                    st.error(f"Search failed: {error_msg}")
                    print(f"DEBUG: API call failed: {error_msg}")

        # Display results from session state
        if st.session_state.home_search_results is not None:
            search_results = st.session_state.home_search_results
            if search_results:
                st.success(f"Found {len(search_results)} matching programs:")

                # Simple text display
                results_text = ""
                for i, program in enumerate(search_results[:10], 1):
                    results_text += f"{i}. {program['program_name']} at {program['college']}\n"
                    results_text += f"Degree: {program['degree_type']}\n"
                    results_text += f"TAP Eligible: {program['tap_eligible']}\n\n"

                st.text_area("Search Results", value=results_text, height=250)
            else:
                st.warning("No matching programs found. Try different search terms.")
    
    st.subheader("Popular CUNY Majors")
    majors_result = call_api("/api/majors")
    if majors_result.get('success'):
        majors = majors_result.get('majors', [])
        cols = st.columns(3)
        for i, major in enumerate(majors[:9]):
            with cols[i % 3]:
                with st.expander(major['Major']):
                    st.write("**Available at:**")
                    for school in major.get('Common Schools', []):
                        st.write(f"• {school}")

    # RAG removed from Home per request; use the Advisor tab instead.

def program_search_page():
    # Program search functionality
    st.header("Program Search")
    st.write("Search through all CUNY programs to find what interests you.")
    # Tip removed per request
    
    st.markdown('<div class="app-card-start"></div>', unsafe_allow_html=True)
    with st.container():
        # Initialize widget state defaults
        if 'ps_query' not in st.session_state:
            st.session_state.ps_query = ""
        if 'ps_degree_type' not in st.session_state:
            st.session_state.ps_degree_type = "Any"
        if 'ps_college_contains' not in st.session_state:
            st.session_state.ps_college_contains = ""
        if 'ps_tap_only' not in st.session_state:
            st.session_state.ps_tap_only = False
        if 'ps_max_results' not in st.session_state:
            st.session_state.ps_max_results = 200
        if 'ps_page' not in st.session_state:
            st.session_state.ps_page = 0
        if 'ps_rows_per_page' not in st.session_state:
            st.session_state.ps_rows_per_page = 10

        search_query = st.text_input(
            "Enter search terms (e.g., 'computer science', 'psychology')",
            key="ps_query",
            placeholder="e.g., psychology queens, computer science"
        )

        # Filters
        with st.expander("Filters"):
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            with c1:
                degree_type = st.selectbox(
                    "Degree type",
                    ["Any", "Associate", "Bachelor", "Certificate"],
                    index=["Any", "Associate", "Bachelor", "Certificate"].index(st.session_state.ps_degree_type)
                    if st.session_state.get("ps_degree_type") in ["Any", "Associate", "Bachelor", "Certificate"] else 0,
                    key="ps_degree_type"
                )
            with c2:
                college_contains = st.text_input(
                    "College contains",
                    key="ps_college_contains",
                    placeholder="e.g., queens, brooklyn"
                )
            with c3:
                tap_only = st.checkbox("TAP eligible only", key="ps_tap_only", value=st.session_state.ps_tap_only)
            with c4:
                max_results = st.number_input(
                    "Max results",
                    min_value=10,
                    max_value=1000,
                    value=st.session_state.ps_max_results,
                    step=10,
                    key="ps_max_results"
                )

        # Initialize session state for search results
        if 'search_results' not in st.session_state:
            st.session_state.search_results = None
        if 'search_query' not in st.session_state:
            st.session_state.search_query = ""

        col_search, col_reset = st.columns([1, 1])
        do_search = False
        with col_search:
            if st.button("Search Programs") and search_query:
                do_search = True
        with col_reset:
            if st.button("Reset filters"):
                st.session_state.ps_query = ""
                st.session_state.ps_degree_type = "Any"
                st.session_state.ps_college_contains = ""
                st.session_state.ps_tap_only = False
                st.session_state.ps_max_results = 200
                st.session_state.ps_rows_per_page = 10
                st.session_state.ps_page = 0
                st.session_state.search_results = None
                st.session_state.search_query = ""
                st.rerun()

        if do_search:
            print(f"DEBUG: Search Programs button clicked with query: '{search_query}'")
            with st.spinner("Searching..."):
                search_data = {"query": search_query, "n_results": int(max_results)}
                print(f"DEBUG: About to call API with search_data: {search_data}")
                result = call_api("/api/search", "POST", search_data)
                print(f"DEBUG: API call result: {result}")
                if result.get("success"):
                    results = result.get("results", [])
                    # Client-side filtering
                    filtered = []
                    for r in results:
                        if degree_type != "Any" and r.get("degree_type") != degree_type:
                            continue
                        if college_contains:
                            if college_contains.lower() not in (r.get("college", "").lower()):
                                continue
                        if tap_only and str(r.get("tap_eligible", "")).lower() not in ["yes", "true", "1"]:
                            continue
                        filtered.append(r)
                    st.session_state.search_results = filtered
                    st.session_state.search_query = search_query
                    st.session_state.ps_page = 0
                    print(f"DEBUG: Stored {len(filtered)} filtered results in session state")
                else:
                    st.session_state.search_results = []
                    st.error(f"Search failed: {result.get('error')}")
                    print(f"DEBUG: API call failed: {result.get('error')}")

        # Display results from session state
        if st.session_state.search_results is not None:
            results = st.session_state.search_results
            if results:
                st.success(f"Found {len(results)} matching programs for '{st.session_state.search_query}'")

                # Pagination controls (top-right)
                top_left, top_right = st.columns([3, 1])
                with top_right:
                    rows_per_page = st.selectbox(
                        "Rows/page",
                        options=[10, 20, 50, 100],
                        index=[10, 20, 50, 100].index(st.session_state.ps_rows_per_page)
                        if st.session_state.get("ps_rows_per_page") in [10, 20, 50, 100] else 0,
                        key="ps_rows_per_page"
                    )

                # Build DataFrame for display/download
                df = pd.DataFrame([
                    {
                        "Rank": r.get("rank", i + 1),
                        "Program": r.get("program_name"),
                        "College": r.get("college"),
                        "Degree": r.get("degree_type"),
                        "TAP Eligible": str(r.get("tap_eligible", "")).title(),
                        "CIP Title": r.get("cip_title"),
                    }
                    for i, r in enumerate(results)
                ])

                # Compute paging window
                total = len(df)
                per_page = int(st.session_state.ps_rows_per_page)
                total_pages = max(1, (total + per_page - 1) // per_page)
                st.session_state.ps_page = min(st.session_state.ps_page, total_pages - 1)
                st.session_state.ps_page = max(0, st.session_state.ps_page)
                start = st.session_state.ps_page * per_page
                end = start + per_page
                page_df = df.iloc[start:end]

                st.dataframe(page_df, use_container_width=True, hide_index=True)

                # Pagination buttons + page indicator
                nav1, nav2, nav3 = st.columns([1, 2, 1])
                with nav1:
                    if st.button("◀ Prev", disabled=st.session_state.ps_page <= 0):
                        st.session_state.ps_page -= 1
                        st.rerun()
                with nav2:
                    st.markdown(
                        f"<div style='text-align:center;color:#6B7280'>Page {st.session_state.ps_page + 1} of {total_pages}</div>",
                        unsafe_allow_html=True,
                    )
                with nav3:
                    if st.button("Next ▶", disabled=st.session_state.ps_page >= total_pages - 1):
                        st.session_state.ps_page += 1
                        st.rerun()

                # CSV download for full filtered results
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download CSV ({len(df)} rows)",
                    data=csv,
                    file_name="cuny_program_search_results.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No matching programs found. Try different search terms.")

def cost_calculator_page():
    # Cost calculator page
    st.header("Cost Calculator")
    st.write("Estimate the annual and total cost based on campus, years, and major area.")

    import json
    data_file = Path(__file__).parent.parent / "data" / "CUNY_Cost_Data.json"
    tuition_rows = []
    extras = []
    if data_file.exists():
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            tuition_rows = cdata.get("TuitionAndFees", [])
            extras = cdata.get("MajorSpecificCosts", [])
        except Exception:
            pass

    col1, col2 = st.columns(2)
    with col1:
        campus = st.selectbox(
            "Campus",
            [r["Campus"] for r in tuition_rows] if tuition_rows else ["City College (CCNY)"]
        )
        years = st.number_input("Years of study", min_value=1, max_value=6, value=4)
        living = st.number_input("Estimated living expenses per year ($)", min_value=0, value=12000, step=500)
    with col2:
        major_type = st.selectbox(
            "Major area (for extra costs)",
            [r["Major Type"] for r in extras] if extras else ["Computer Science / Data Science"]
        )
        aid_offset = st.number_input("Estimated annual aid/scholarships ($)", min_value=0, value=3000, step=500)

    # Parse base tuition number from string like "$7,560"
    def _parse_money(s: str) -> int:
        try:
            return int(str(s).replace("$", "").replace(",", "").split(" ")[0])
        except Exception:
            return 0

    base = next((r for r in tuition_rows if r.get("Campus") == campus), None)
    base_tuition = _parse_money(base.get("Annual Tuition & Fees (In-State)", "$7,560")) if base else 7560
    extra_row = next((r for r in extras if r.get("Major Type") == major_type), None)
    extra_est = _parse_money(extra_row.get("Estimated Extra Costs / Year", "$0")) if extra_row else 0

    annual_cost = max(0, base_tuition + living + extra_est - aid_offset)
    total_cost = annual_cost * years

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Base Tuition & Fees (annual)", f"${base_tuition:,.0f}")
    m2.metric("Estimated Extras (annual)", f"${extra_est:,.0f}")
    m3.metric("Living (annual)", f"${living:,.0f}")
    st.metric("Estimated Annual Out-of-Pocket", f"${annual_cost:,.0f}")
    st.metric("Estimated Total (program)", f"${total_cost:,.0f}")

    with st.expander("How these are calculated"):
        st.write("Annual = Tuition + Living + Major Extras - Aid; Total = Annual × Years.")

def compare_programs_page():
    # Program comparison page
    st.header("Compare Programs")
    st.write("Pick two programs to compare key details side-by-side.")

    # Simple selection by querying and letting user pick
    q = st.text_input("Search terms (used for both pickers)", placeholder="e.g., computer science queens")
    if q:
        result = call_api("/api/search", "POST", {"query": q, "n_results": 50})
        options = result.get("results", []) if result.get("success") else []
    else:
        options = []

    def _opt_label(r: Dict[str, Any]) -> str:
        return f"{r.get('program_name')} — {r.get('college')} ({r.get('degree_type')})"

    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Program A", options, format_func=_opt_label, index=0 if options else None)
    with c2:
        b = st.selectbox("Program B", options, format_func=_opt_label, index=1 if len(options) > 1 else (0 if options else None))

    if options and a and b:
        import pandas as _pd
        from datetime import datetime
        import json as _json
        import re as _re
        # Date helpers
        def _fmt_date(d: str) -> str:
            try:
                return str(d)[:10] if d else ""
            except Exception:
                return str(d or "")
        def _age_years(d: str) -> str:
            try:
                if not d:
                    return ""
                dt = datetime.fromisoformat(str(d).replace("Z", "").split(".")[0])
                years = (datetime.now() - dt).days // 365
                return f"{years}"
            except Exception:
                return ""
        # Money and mapping helpers
        def _parse_money_any(s: str) -> int:
            try:
                if not s:
                    return 0
                nums = _re.findall(r"[0-9][0-9,]*", str(s))
                vals = [int(n.replace(",", "")) for n in nums]
                if not vals:
                    return 0
                return int(sum((vals[0], vals[-1])) / 2) if len(vals) > 1 else vals[0]
            except Exception:
                return 0
        def _normalize_campus(college_name: str) -> str:
            name = (college_name or "").strip()
            mapping = {
                "City College": "City College (CCNY)",
                "Cuny City College": "City College (CCNY)",
                "New York City College of Technology": "City Tech (NYCCT)",
                "NYC College of Technology": "City Tech (NYCCT)",
                "Nyc College Of Technology": "City Tech (NYCCT)",
            }
            return mapping.get(name, name)
        # Load cost data
        tuition_rows, extras_rows = [], []
        try:
            cpath = Path(__file__).parent.parent / "data" / "CUNY_Cost_Data.json"
            if cpath.exists():
                with open(cpath, "r", encoding="utf-8") as f:
                    cdata = _json.load(f)
                tuition_rows = cdata.get("TuitionAndFees", [])
                extras_rows = cdata.get("MajorSpecificCosts", [])
        except Exception:
            pass
        def _get_tuition_str(college_name: str) -> str:
            campus = _normalize_campus(college_name)
            row = next((r for r in tuition_rows if str(r.get("Campus", "")).strip().lower() == campus.lower()), None)
            if not row:
                low = campus.lower()
                row = next((r for r in tuition_rows if low in str(r.get("Campus", "")).lower()), None)
            return row.get("Annual Tuition & Fees (In-State)") if row else "N/A"
        def _guess_major_type(prog: Dict[str, Any]) -> str:
            text = f"{prog.get('program_name','')} {prog.get('cip_title','')} {prog.get('degree_type','')}".lower()
            if any(k in text for k in ["computer science", "data science", "informatics"]):
                return "Computer Science / Data Science"
            if "engineering" in text or "tech" in text:
                return "Engineering / Tech Labs"
            if any(k in text for k in ["biology", "neuroscience", "life science", "pre-med", "pre med"]):
                return "Pre‑Med / Life Sciences"
            if any(k in text for k in ["criminal justice", "public policy", "pre-law", "pre law", "political science"]):
                return "Pre‑Law / Public Policy"
            if "education" in text or "teacher" in text:
                return "Education / Teaching"
            if any(k in text for k in ["business", "accounting", "finance", "marketing", "economics", "econ"]):
                return "Business / Econ"
            if any(k in text for k in ["studio", "fine art", "graphic design", "design"]):
                return "Studio & Fine Arts"
            return "(not inferred)"
        def _extras_str_for(major_type: str) -> str:
            if not major_type or major_type.startswith("("):
                return "$0"
            row = next((r for r in extras_rows if str(r.get("Major Type", "")).lower() == major_type.lower()), None)
            return row.get("Estimated Extra Costs / Year", "$0") if row else "$0"

        # Cost controls
        col_l, col_a = st.columns(2)
        with col_l:
            living = st.number_input("Assumed living (annual) for both ($)", min_value=0, value=12000, step=500, key="compare_living")
        with col_a:
            aid = st.number_input("Assumed annual aid/scholarships ($)", min_value=0, value=3000, step=500, key="compare_aid")

        # Compute cost data for A and B
        tuition_str_a = _get_tuition_str(a.get("college"))
        tuition_str_b = _get_tuition_str(b.get("college"))
        tuition_a = _parse_money_any(tuition_str_a)
        tuition_b = _parse_money_any(tuition_str_b)
        major_type_a = _guess_major_type(a)
        major_type_b = _guess_major_type(b)
        extras_str_a = _extras_str_for(major_type_a)
        extras_str_b = _extras_str_for(major_type_b)
        extras_a = _parse_money_any(extras_str_a)
        extras_b = _parse_money_any(extras_str_b)
        oop_a = max(0, tuition_a + living + extras_a - aid)
        oop_b = max(0, tuition_b + living + extras_b - aid)

        rows = [
            {"Field": "Program", "A": a.get("program_name"), "B": b.get("program_name")},
            {"Field": "College", "A": a.get("college"), "B": b.get("college")},
            {"Field": "Degree", "A": a.get("degree_type"), "B": b.get("degree_type")},
            {"Field": "CIP Title", "A": a.get("cip_title"), "B": b.get("cip_title")},
            {"Field": "CIP Code", "A": a.get("cip_code"), "B": b.get("cip_code")},
            {"Field": "IRP Code", "A": a.get("irp_code"), "B": b.get("irp_code")},
            {"Field": "Date Established", "A": _fmt_date(a.get("date_program_established")), "B": _fmt_date(b.get("date_program_established"))},
            {"Field": "Program Age (yrs)", "A": _age_years(a.get("date_program_established")), "B": _age_years(b.get("date_program_established"))},
            {"Field": "Record Type", "A": a.get("record_type"), "B": b.get("record_type")},
            {"Field": "Period", "A": a.get("period"), "B": b.get("period")},
            {"Field": "TAP Eligible", "A": str(a.get("tap_eligible", "")).title(), "B": str(b.get("tap_eligible", "")).title()},
            # Cost comparisons
            {"Field": "Annual Tuition & Fees (In-State)", "A": tuition_str_a, "B": tuition_str_b},
            {"Field": "Major Type (for extras)", "A": major_type_a, "B": major_type_b},
            {"Field": "Estimated Extras (annual)", "A": extras_str_a, "B": extras_str_b},
            {"Field": "Assumed Living (annual)", "A": f"${living:,.0f}", "B": f"${living:,.0f}"},
            {"Field": "Est. Annual Out-of-Pocket", "A": f"${oop_a:,.0f}", "B": f"${oop_b:,.0f}"},
        ]
        # Differences only toggle
        diff_only = st.checkbox("Differences only", value=False)
        if diff_only:
            rows = [r for r in rows if (str(r.get("A")) != str(r.get("B")))]
        df = _pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        # Export
        st.download_button(
            "Download comparison CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="program_comparison.csv",
            mime="text/csv",
        )

def advisor_page():
    # Dedicated Advisor page to make RAG a first-class feature
    st.header("Ask the Advisor (RAG)")
    st.markdown('<div class="app-card-start"></div>', unsafe_allow_html=True)
    # Toggle for using quiz profile
    st.checkbox("Use my quiz profile when answering", key="advisor_use_quiz_profile", value=st.session_state.get("advisor_use_quiz_profile", True))
    with st.container():
        ask_advisor_ui(key_prefix="advisor", use_quiz_profile=st.session_state.get("advisor_use_quiz_profile", True))

    # Recent history
    st.markdown("---")
    st.subheader("Recent Questions")
    try:
        user_id = _get_user_id()
        clear_col, _ = st.columns([1, 5])
        with clear_col:
            if st.button("Clear my history", use_container_width=False):
                try:
                    _history.clear_user(user_id)
                    st.success("History cleared.")
                except Exception as e:
                    st.error(f"Failed to clear: {e}")
        history = _history.get_recent_qa(user_id, limit=20)
        if history:
            buf = []
            for item in history:
                who = "You" if item.role == "user" else "Advisor"
                buf.append(f"- **{who}:** {item.content}")
            st.markdown("\n".join(buf))
            # Export CSV
            try:
                from datetime import datetime
                rows = [
                    {
                        "timestamp": datetime.fromtimestamp(h.ts).isoformat(),
                        "role": h.role,
                        "content": h.content,
                        "session_id": h.session_id,
                    }
                    for h in history
                ]
                import pandas as _pd
                csv_bytes = _pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download my Q&A history ({len(rows)} rows)",
                    data=csv_bytes,
                    file_name="advisor_history.csv",
                    mime="text/csv",
                )
            except Exception:
                pass
        else:
            st.info("No history yet. Ask your first question above.")
    except Exception as e:
        st.warning(f"Could not load history: {e}")
    # Show completion status and full recommendations outside of any form
    if hasattr(st.session_state, 'quiz_completed') and st.session_state.quiz_completed:
        st.success("🎉 Quiz completed! Here are your personalized recommendations.")

        # Render all recommendations (moved from old Recommendations page)
        if st.session_state.recommendations:
            for i, rec in enumerate(st.session_state.recommendations, 1):
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h3>{i}. {rec['program_name']}</h3>
                        <h4>{rec['college']}</h4>
                        <p><strong>Degree:</strong> {rec['degree_type']}</p>
                        <p><strong>Estimated Annual Cost:</strong> ${rec['estimated_cost']:,}</p>
                        <p><span class="fit-score">Fit Score: {rec['fit_score']:.2f}/1.0</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                with st.expander(f"Details for {rec['program_name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Why This Program:** {rec.get('reasoning')}")
                        if rec.get('pros'):
                            st.markdown("**Pros:**")
                            for pro in rec.get('pros', []):
                                st.markdown(f"• {pro}")
                    with col2:
                        fit_score = rec.get('fit_score', 0)
                        st.markdown(f"**Fit Score:** {fit_score:.2f}/1.0")
                        st.progress(fit_score)

    # (Removed) Ask the Advisor duplication here to avoid duplicate widget keys with the Quiz page

    # RAG moved to Home and Quiz completion to meet UX intent

def quiz_page():
    # Comprehensive quiz page to help students decide on their major
    st.header("CUNY Major Selection Quiz")
    st.markdown("*This questionnaire will help you discover the right CUNY program based on your interests, goals, and circumstances.* 🧭")
    
    # Initialize session state for quiz progress
    if 'quiz_progress' not in st.session_state:
        st.session_state.quiz_progress = 0
    if 'quiz_responses' not in st.session_state:
        st.session_state.quiz_responses = {}
    
    # Progress bar
    progress = (st.session_state.quiz_progress + 1) / 6
    st.progress(progress)
    st.write(f"Progress: {st.session_state.quiz_progress + 1}/6 sections completed")
    
    # Quiz sections based on your detailed specification
    sections = [
        "Getting Started",
        "Academic Interests & Passions", 
        "Strengths & Abilities",
        "Career Values & Goals",
        "Financial Considerations",
        "CUNY System Navigation"
    ]
    
    # Section navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Previous") and st.session_state.quiz_progress > 0:
            st.session_state.quiz_progress -= 1
            st.rerun()
    
    with col2:
        st.write(f"**Section {st.session_state.quiz_progress + 1}: {sections[st.session_state.quiz_progress]}**")
    
    with col3:
        if st.button("Next →") and st.session_state.quiz_progress < 5:
            st.session_state.quiz_progress += 1
            st.rerun()
    
    st.markdown("---")
    
    # Section content
    if st.session_state.quiz_progress == 0:
        # Getting Started
        st.subheader("Getting Started")
        
        with st.form("section_1"):
            st.markdown("**Let's begin with some basic information about you:**")
            
            # Current status
            current_status = st.selectbox(
                "What's your current educational status?",
                ["High School Senior", "Recent High School Graduate", "Community College Student", 
                 "Transfer Student", "Returning Adult Student", "Currently Enrolled at CUNY", "Other"]
            )
            
            # Decision timeline
            decision_timeline = st.selectbox(
                "When are you planning to start or change your program?",
                ["This Fall", "Next Spring", "Next Fall", "Within 2 years", "Just exploring options"]
            )
            
            # Certainty level
            certainty = st.slider(
                "How certain are you about your major choice?",
                0, 10, 5,
                help="0 = Completely unsure, 10 = Very certain"
            )
            
            # Previous considerations
            previous_majors = st.multiselect(
                "Have you previously considered any of these fields?",
                ["Business", "Engineering", "Liberal Arts", "Sciences", "Education", 
                 "Healthcare", "Computer Science", "Psychology", "Criminal Justice", 
                 "Communications", "Art & Design", "Social Work", "Other"]
            )
            
            # Influences
            influences = st.multiselect(
                "What factors are most influencing your decision? (Select all that apply)",
                ["Family expectations", "Personal interests", "Job market prospects", 
                 "Salary potential", "Social impact", "Personal experience", 
                 "Academic strengths", "Media/pop culture", "Career counselor advice", "Peer influence"]
            )
            
            if st.form_submit_button("Save & Continue", type="primary"):
                st.session_state.quiz_responses['section_1'] = {
                    'current_status': current_status,
                    'decision_timeline': decision_timeline,
                    'certainty': certainty,
                    'previous_majors': previous_majors,
                    'influences': influences
                }
                st.session_state.quiz_progress = 1
                st.success("Section 1 completed!")
                st.rerun()
    
    elif st.session_state.quiz_progress == 1:
        # Academic Interests & Passions
        st.subheader("Academic Interests & Passions")
        
        with st.form("section_2"):
            st.markdown("**Tell us about your academic interests:**")
            
            # Subject preferences
            favorite_subjects = st.multiselect(
                "Which high school subjects did you enjoy most?",
                ["Mathematics", "English/Literature", "History", "Science (Biology)", 
                 "Science (Chemistry)", "Science (Physics)", "Foreign Languages", 
                 "Art", "Music", "Computer Science", "Social Studies", "Psychology", 
                 "Economics", "Physical Education", "Business", "Other"]
            )
            
            # Learning style
            learning_style = st.selectbox(
                "How do you learn best?",
                ["Hands-on activities and experiments", "Reading and writing", 
                 "Visual aids and diagrams", "Group discussions", 
                 "Independent study", "Online/digital learning", "Traditional lectures"]
            )
            
            # Problem-solving approach
            problem_solving = st.selectbox(
                "When facing a complex problem, you prefer to:",
                ["Break it down systematically", "Use creative/artistic approaches", 
                 "Collaborate with others", "Research extensively first", 
                 "Jump in and learn by doing", "Seek expert guidance", "Use technology/tools"]
            )
            
            # Passion projects
            passion_areas = st.multiselect(
                "What topics do you find yourself reading about or discussing in your free time?",
                ["Technology and innovation", "Social justice and politics", "Health and wellness", 
                 "Environment and sustainability", "Business and entrepreneurship", 
                 "Arts and culture", "Sports and fitness", "Travel and cultures", 
                 "Science and research", "Education and teaching", "Media and entertainment"]
            )
            
            # Activity preferences
            activity_prefs = st.multiselect(
                "Which activities energize you most?",
                ["Solving puzzles or problems", "Creating art or content", "Helping others", 
                 "Leading teams or projects", "Analyzing data or information", 
                 "Building or designing things", "Writing or communicating", 
                 "Performing or presenting", "Researching or investigating"]
            )
            
            if st.form_submit_button("Save & Continue", type="primary"):
                st.session_state.quiz_responses['section_2'] = {
                    'favorite_subjects': favorite_subjects,
                    'learning_style': learning_style,
                    'problem_solving': problem_solving,
                    'passion_areas': passion_areas,
                    'activity_prefs': activity_prefs
                }
                st.session_state.quiz_progress = 2
                st.success("Section 2 completed!")
                st.rerun()
    
    elif st.session_state.quiz_progress == 2:
        # Strengths & Abilities
        st.subheader("Strengths & Abilities")
        
        with st.form("section_3"):
            st.markdown("**Let's identify your natural strengths:**")
            
            # Academic strengths
            academic_strengths = st.multiselect(
                "In which areas do you consistently perform well?",
                ["Mathematical reasoning", "Written communication", "Verbal communication", 
                 "Scientific thinking", "Creative expression", "Critical analysis", 
                 "Research skills", "Problem-solving", "Leadership", "Teamwork", 
                 "Technical skills", "Organization", "Time management"]
            )
            
            # Skills confidence
            st.markdown("**Rate your confidence in these skill areas (1-5 scale):**")
            
            col1, col2 = st.columns(2)
            with col1:
                math_confidence = st.slider("Mathematics", 1, 5, 3)
                writing_confidence = st.slider("Writing", 1, 5, 3)
                public_speaking = st.slider("Public Speaking", 1, 5, 3)
                tech_skills = st.slider("Technology", 1, 5, 3)
            
            with col2:
                leadership_confidence = st.slider("Leadership", 1, 5, 3)
                creativity_confidence = st.slider("Creativity", 1, 5, 3)
                analysis_confidence = st.slider("Analysis", 1, 5, 3)
                collaboration = st.slider("Collaboration", 1, 5, 3)

            # Additional preferences
            work_style = st.selectbox(
                "Which work style fits you best?",
                [
                    "Working independently with minimal supervision",
                    "Collaborating closely with a small team",
                    "Leading and directing others",
                    "Following clear instructions and procedures",
                ],
            )
            challenge_pref = st.slider("How much challenge do you prefer in coursework?", 1, 5, 3)

            if st.form_submit_button("Save & Continue", type="primary"):
                st.session_state.quiz_responses['section_3'] = {
                    'academic_strengths': academic_strengths,
                    'confidence_ratings': {
                        'math': math_confidence,
                        'writing': writing_confidence,
                        'speaking': public_speaking,
                        'tech': tech_skills,
                        'leadership': leadership_confidence,
                        'creativity': creativity_confidence,
                        'analysis': analysis_confidence,
                        'collaboration': collaboration,
                    },
                    'work_style': work_style,
                    'challenge_pref': challenge_pref,
                }
                st.session_state.quiz_progress = 3
                st.success("Section 3 completed!")
                st.rerun()
    
    elif st.session_state.quiz_progress == 3:
        # Career Values & Goals
        st.subheader("Career Values & Goals")
        
        with st.form("section_4"):
            st.markdown("**What matters most to you in your future career?**")
            
            # Core values
            core_values = st.multiselect(
                "Select your top career values:",
                ["High salary potential", "Job security and stability", "Work-life balance", 
                 "Making a positive social impact", "Creative expression", "Intellectual challenge", 
                 "Leadership opportunities", "Flexibility and autonomy", "Prestige and recognition", 
                 "Helping others directly", "Innovation and cutting-edge work", 
                 "Teamwork and collaboration", "Travel opportunities", "Entrepreneurship potential"]
            )
            
            # Work environment
            work_environment = st.selectbox(
                "What work environment appeals to you most?",
                ["Corporate office setting", "Healthcare facilities", "Educational institutions", 
                 "Creative studios/agencies", "Research laboratories", "Outdoor/field work", 
                 "Government agencies", "Non-profit organizations", "Tech companies", 
                 "Small business/startup", "Remote/work from home", "Mixed/flexible environments"]
            )
            
            # Career trajectory
            career_goal = st.selectbox(
                "What's your long-term career vision?",
                ["Becoming an expert/specialist in my field", "Moving into leadership/management", 
                 "Starting my own business", "Making significant social impact", 
                 "Achieving financial independence", "Balancing career with personal life", 
                 "Continuous learning and growth", "Becoming recognized in my profession"]
            )
            
            # Industry interests
            industry_interests = st.multiselect(
                "Which industries interest you most?",
                ["Technology", "Healthcare", "Education", "Finance", "Media/Entertainment", 
                 "Government/Public Service", "Non-profit/Social Services", "Business/Consulting", 
                 "Arts/Creative", "Science/Research", "Engineering", "Law/Legal", 
                 "Marketing/Advertising", "Real Estate", "Other"]
            )
            
            # Success definition
            success_definition = st.text_area(
                "How do you define success in your career?",
                placeholder="Describe what success means to you personally..."
            )
            
            if st.form_submit_button("Save & Continue", type="primary"):
                st.session_state.quiz_responses['section_4'] = {
                    'core_values': core_values,
                    'work_environment': work_environment,
                    'career_goal': career_goal,
                    'industry_interests': industry_interests,
                    'success_definition': success_definition
                }
                st.session_state.quiz_progress = 4
                st.success("Section 4 completed!")
                st.rerun()
    
    elif st.session_state.quiz_progress == 4:
        # Financial Considerations
        st.subheader("Financial Considerations")
        
        with st.form("section_5"):
            st.markdown("**Let's discuss the financial aspects of your education:**")
            
            # Budget constraints
            tuition_budget = st.selectbox(
                "What's your annual tuition budget?",
                ["Under $3,000 (In-state community college)", 
                 "$3,000 - $7,000 (CUNY senior colleges)", 
                 "$7,000 - $15,000 (Private or out-of-state)", 
                 "$15,000+ (No budget constraints)", 
                 "Depends on financial aid"]
            )
            
            # Financial aid
            financial_aid_status = st.multiselect(
                "What financial aid are you eligible for or considering?",
                ["Pell Grant", "TAP (Tuition Assistance Program)", "CUNY scholarships", 
                 "Merit-based scholarships", "Need-based aid", "Work-study programs", 
                 "Student loans", "Family support", "Personal savings", "Employer assistance"]
            )
            
            # ROI considerations
            roi_importance = st.slider(
                "How important is return on investment (ROI) in your program choice?",
                1, 10, 5,
                help="1 = Not important, 10 = Extremely important"
            )
            
            # Debt comfort
            debt_comfort = st.selectbox(
                "How comfortable are you with taking on student debt?",
                ["Very comfortable - education is an investment", 
                 "Somewhat comfortable - minimal debt only", 
                 "Uncomfortable - prefer to avoid debt", 
                 "Will only consider debt-free options"]
            )
            
            # Time to degree
            time_preference = st.selectbox(
                "What's your preferred timeline to complete your degree?",
                ["2 years (Associate)", "4 years (Bachelor's - traditional)", 
                 "5-6 years (Bachelor's - part-time)", "Flexible - as long as needed", 
                 "Accelerated programs preferred"]
            )
            
            # Employment during school
            work_plan = st.selectbox(
                "Do you plan to work while attending school?",
                ["Full-time work, part-time school", "Part-time work, full-time school", 
                 "No work, focus on studies", "Work-study programs only", 
                 "Flexible depending on opportunities"]
            )
            
            if st.form_submit_button("Save & Continue", type="primary"):
                st.session_state.quiz_responses['section_5'] = {
                    'tuition_budget': tuition_budget,
                    'financial_aid_status': financial_aid_status,
                    'roi_importance': roi_importance,
                    'debt_comfort': debt_comfort,
                    'time_preference': time_preference,
                    'work_plan': work_plan
                }
                st.session_state.quiz_progress = 5
                st.success("Section 5 completed!")
                st.rerun()
    
    elif st.session_state.quiz_progress == 5:
        # CUNY System Navigation
        st.subheader("CUNY System Navigation")
        
        with st.form("section_6"):
            st.markdown("**Let's find the best CUNY options for you:**")
            
            # Location preferences
            location_pref = st.multiselect(
                "Which CUNY locations are convenient for you?",
                ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", 
                 "No preference - willing to travel", "Online/hybrid options preferred"]
            )
            
            # Campus size preference
            campus_size = st.selectbox(
                "What campus size do you prefer?",
                ["Large university (10,000+ students)", "Medium college (5,000-10,000 students)", 
                 "Small college (Under 5,000 students)", "No preference"]
            )
            
            # Program delivery
            delivery_pref = st.selectbox(
                "How do you prefer to take classes?",
                ["Fully in-person", "Fully online", "Hybrid (mix of online and in-person)", 
                 "Evening classes", "Weekend classes", "Flexible/no preference"]
            )
            
            # CUNY knowledge
            cuny_familiarity = st.slider(
                "How familiar are you with the CUNY system?",
                1, 10, 5,
                help="1 = Never heard of it, 10 = Very familiar"
            )
            
            # Support services
            support_services = st.multiselect(
                "Which support services are important to you?",
                ["Academic tutoring", "Career counseling", "Financial aid guidance", 
                 "Mental health services", "Disability services", "ESL support", 
                 "Transfer credit assistance", "Internship placement", "Job placement", 
                 "Study abroad programs", "Research opportunities"]
            )
            
            # Special programs
            special_interests = st.multiselect(
                "Are you interested in any of these special programs?",
                ["Honors programs", "Pre-professional tracks (pre-med, pre-law)", 
                 "Accelerated degree programs", "Dual degree programs", 
                 "Study abroad partnerships", "Cooperative education (co-op)", 
                 "Research opportunities", "Community service learning"]
            )
            
            if st.form_submit_button("Complete Quiz & Get Recommendations", type="primary"):
                st.session_state.quiz_responses['section_6'] = {
                    'location_pref': location_pref,
                    'campus_size': campus_size,
                    'delivery_pref': delivery_pref,
                    'cuny_familiarity': cuny_familiarity,
                    'support_services': support_services,
                    'special_interests': special_interests
                }
                
                # Process quiz results
                with st.spinner("Analyzing your responses and generating personalized recommendations..."):
                    # Transform quiz responses into recommendation format
                    quiz_data = {
                        'quiz_responses': st.session_state.quiz_responses,
                        'interests': st.session_state.quiz_responses.get('section_2', {}).get('favorite_subjects', []),
                        'values': st.session_state.quiz_responses.get('section_4', {}).get('core_values', []),
                        'budget_info': st.session_state.quiz_responses.get('section_5', {}).get('tuition_budget', ''),
                        'location_pref': st.session_state.quiz_responses.get('section_6', {}).get('location_pref', [])
                    }
                    
                    # Call recommendations API
                    result = call_api("/api/recommendations", "POST", quiz_data)
                    
                    if result.get('success'):
                        st.session_state.recommendations = result.get('recommendations', [])
                        st.session_state.quiz_completed = True
                        st.success(f"Quiz completed! Found {len(st.session_state.recommendations)} personalized recommendations based on your responses.")
                        st.balloons()
                        
                        # Show recommendations immediately
                        st.subheader("Your Personalized Recommendations")
                        
                        for i, rec in enumerate(st.session_state.recommendations[:3], 1):  # Show top 3
                            with st.container():
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h3>{i}. {rec['program_name']}</h3>
                                    <h4>{rec['college']}</h4>
                                    <p><strong>Degree:</strong> {rec['degree_type']}</p>
                                    <p><strong>Estimated Annual Cost:</strong> ${rec['estimated_cost']:,}</p>
                                    <p><span class="fit-score">Fit Score: {rec['fit_score']:.2f}/1.0</span></p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    else:
                        st.error(f"❌ Error generating recommendations: {result.get('error', 'Unknown error')}")
    
    # Show completion status and additional actions outside of any form
    if hasattr(st.session_state, 'quiz_completed') and st.session_state.quiz_completed:
        st.success("🎉 Quiz completed! Your recommendations are shown above.")

        # Ask the Advisor right after quiz completion
        st.markdown("---")
        st.subheader("Ask the Advisor (RAG)")
        st.markdown('<div class="app-card-start"></div>', unsafe_allow_html=True)
        with st.container():
            ask_advisor_ui(key_prefix="quiz", use_quiz_profile=True)
        
    # Removed old navigation to Recommendations tab
    
    # Reset quiz option
    if st.session_state.quiz_progress > 0:
        st.markdown("---")
    if st.button("Restart Quiz"):
            st.session_state.quiz_progress = 0
            st.session_state.quiz_responses = {}
            if 'quiz_completed' in st.session_state:
                del st.session_state.quiz_completed
            st.rerun()

def main():
    """
    🎯 Main Streamlit application with tabbed interface
    """
    
    # Header
    display_header()
    
    # Initialize session state
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {}
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Navigation
    tabs = display_navigation()
    
    # Tab content
    with tabs[0]:  # Home
        home_page()
    
    with tabs[1]:  # Major Discovery Quiz
        quiz_page()
    
    with tabs[2]:  # Advisor (RAG)
        advisor_page()

    with tabs[3]:  # Program Search
        program_search_page()
    
    with tabs[4]:  # Cost Calculator
        cost_calculator_page()
    
    with tabs[5]:  # Compare Programs
        compare_programs_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
    <p>Dream 4 Degree | Built with ❤️ for CUNY Students</p>
        <p>Data sourced from NY.gov and CUNY official resources</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()