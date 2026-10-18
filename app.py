import streamlit as st, time, os
from core.tokenizer import TokenAnalyzer
from core.compressor import FastHeuristicCompressor, AICompressor, MAX_INPUT_LENGTH
from core.cost_estimator import CostEstimator, DEFAULT_PRICING_PER_MILLION

st.set_page_config(page_title="Prompt Token Optimizer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.metric-card { background:#1a1c24; border:1px solid #2e3440; border-radius:10px; padding:16px; text-align:center; box-shadow:0 4px 6px -1px rgba(0,0,0,0.1); }
.metric-value-orig { font-size:28px; font-weight:700; color:#ff5c5c; }
.metric-value-opt { font-size:28px; font-weight:700; color:#00e676; }
.metric-value-saved { font-size:28px; font-weight:700; color:#38bdf8; }
.metric-value-roi { font-size:28px; font-weight:700; color:#a78bfa; }
.metric-label { font-size:13px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-top:4px; }
.stTextArea textarea { font-family:'JetBrains Mono','Fira Code',monospace; font-size:14px; }
</style>""", unsafe_allow_html=True)

def render_card(val, label, style="metric-value-saved"):
    st.markdown(f'<div class="metric-card"><div class="{style}">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

@st.cache_resource
def get_analyzer(): return TokenAnalyzer()
analyzer = get_analyzer()

PRESETS = {
    "PDF Scan Request": "do a pdf scan for me",
    "Sales Data Analysis": "Hey! Could you please help me to do an analysis of this sales data and tell me what is inside it? Thanks in advance!",
    "Database Query Implementation": "Hello, I was wondering if you could write some code to connect to PostgreSQL database and fetch users. Thanks!",
    "Python Bug Fix": "Basically I just want you to figure out why this python script is failing with KeyError",
    "Article Summary": "Can you please kindly do a summary of this article for me asap?"
}

with st.sidebar:
    st.header("Engine Settings")
    mode = st.radio("Compression Strategy", ["Fast Offline (Rule-Based)", "AI-Powered (Semantic Restructuring)"], index=0, help="Fast Offline runs in <5ms without API calls. AI-Powered uses Gemini.")
    gemini_key, allow_env = "", False
    if "AI-Powered" in mode:
        gemini_key = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API key.")
        if bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) and not gemini_key:
            allow_env = st.checkbox("Use host environment key (GEMINI_API_KEY)", value=False, help="Enable only if authorized to consume host quota.")
    st.markdown("---")
    st.subheader("Quick Examples")
    preset = st.selectbox("Load a sample human input:", ["Select an example..."] + list(PRESETS.keys()))

st.title("Prompt Token Optimizer & Analyzer")
st.caption("Convert conversational human requests into token-minimized AI prompts with live token savings.")

default_in = PRESETS.get(preset, "") if preset != "Select an example..." else ""
user_input = st.text_area("Enter Human Request:", value=default_in, height=130, max_chars=MAX_INPUT_LENGTH, placeholder=f"Type naturally. Limit: {MAX_INPUT_LENGTH:,} characters.")

col_run, _ = st.columns([1, 5])
with col_run: optimize_btn = st.button("Optimize Prompt", type="primary", use_container_width=True)

if user_input:
    orig_tokens, char_len = analyzer.count_tokens(user_input), len(user_input)
    if char_len > MAX_INPUT_LENGTH:
        st.error(f"Input length ({char_len:,} characters) exceeds maximum allowed limit of {MAX_INPUT_LENGTH:,} characters.")
    else:
        st.caption(f"Input Length: **{orig_tokens}** tokens | **{char_len:,}** / {MAX_INPUT_LENGTH:,} characters")

if optimize_btn and user_input.strip() and len(user_input) <= MAX_INPUT_LENGTH:
    start_time = time.perf_counter()
    optimized_text, error_msg = "", None

    if "Fast Offline" in mode:
        try: optimized_text = FastHeuristicCompressor.compress(user_input)
        except Exception as e: st.error(f"Compression error: {e}"); optimized_text = user_input
    else:
        ai = AICompressor(api_key=gemini_key.strip() if gemini_key else None, allow_env_fallback=allow_env)
        if not ai.is_available():
            st.warning("No valid Gemini API key configured. Provide an API key or check 'Use host environment key'. Falling back to fast offline engine.")
            optimized_text = FastHeuristicCompressor.compress(user_input)
        else:
            with st.spinner("AI is compressing prompt..."): optimized_text, error_msg = ai.compress(user_input)
            if error_msg:
                st.error(f"AI Compression Notice: {error_msg} Defaulting to offline engine.")
                optimized_text = FastHeuristicCompressor.compress(user_input)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    metrics = analyzer.analyze_comparison(user_input, optimized_text)

    st.markdown("### Token Reduction Dashboard")
    m1, m2, m3, m4 = st.columns(4)
    with m1: render_card(metrics['original_tokens'], "Original Tokens", "metric-value-orig")
    with m2: render_card(metrics['optimized_tokens'], "Optimized Tokens", "metric-value-opt")
    with m3: render_card(f"-{metrics['tokens_saved']} ({metrics['percentage_saved']}%)", "Tokens Saved")
    with m4: render_card(f"{elapsed_ms:.1f} ms", "Processing Time")

    st.markdown("<br>", unsafe_allow_html=True)
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Original Human Input")
        st.text_area("Original", value=user_input, height=160, disabled=True, key="orig_disp")
    with c_right:
        st.subheader("Optimized AI Prompt")
        st.text_area("Optimized", value=optimized_text, height=160, key="opt_disp")
        st.code(optimized_text, language="markdown")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Projected Cost Savings and ROI")
    st.caption("Calculate projected financial savings based on your anticipated API request volumes across leading LLMs.")

    roi_left, roi_right = st.columns([3, 2])
    with roi_left:
        volume = st.select_slider("Monthly Request Volume:", options=[1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000], value=100_000, format_func=lambda x: f"{x:,} requests/month")
    with roi_right:
        target_model = st.selectbox("Primary Benchmark Model:", options=list(DEFAULT_PRICING_PER_MILLION.keys()), index=1)

    roi_data = CostEstimator.estimate_savings(metrics["tokens_saved"], metrics["original_tokens"], metrics["optimized_tokens"], request_volume=volume)
    prim = roi_data["models"][target_model]

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1: render_card(f"${prim['original_cost']:,.2f}", f"Original Cost ({volume:,} reqs)", "metric-value-orig")
    with rc2: render_card(f"${prim['optimized_cost']:,.2f}", f"Optimized Cost ({volume:,} reqs)", "metric-value-opt")
    with rc3: render_card(f"${prim['savings']:,.2f}", f"Net Savings ({target_model})", "metric-value-roi")
    with rc4: render_card(f"${prim['savings_per_1m']:,.2f}", "Savings per 1M Requests")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Model Pricing Comparison Matrix")
    st.dataframe([{
        "Model": name,
        "Rate / 1M Tokens": f"${info['rate_per_million']:.2f}",
        f"Original ({volume:,} reqs)": f"${info['original_cost']:,.2f}",
        f"Optimized ({volume:,} reqs)": f"${info['optimized_cost']:,.2f}",
        f"Net Savings ({volume:,} reqs)": f"${info['savings']:,.2f}",
        "Savings per 1M Reqs": f"${info['savings_per_1m']:,.2f}",
        "Token Reduction": f"{metrics['percentage_saved']}%"
    } for name, info in roi_data["models"].items()], use_container_width=True, hide_index=True)
