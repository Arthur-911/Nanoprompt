import streamlit as st
import time
from core.tokenizer import TokenAnalyzer
from core.compressor import FastHeuristicCompressor, AICompressor

st.set_page_config(
    page_title="Prompt Token Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-performance, modern dark-mode aesthetic
st.markdown("""
<style>
    .metric-card {
        background-color: #1a1c24;
        border: 1px solid #2e3440;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value-orig {
        font-size: 28px;
        font-weight: 700;
        color: #ff5c5c;
    }
    .metric-value-opt {
        font-size: 28px;
        font-weight: 700;
        color: #00e676;
    }
    .metric-value-saved {
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .stTextArea textarea {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Analyzer
@st.cache_resource
def get_analyzer():
    return TokenAnalyzer()

analyzer = get_analyzer()

# Sidebar: Configuration & Presets
with st.sidebar:
    st.header("⚡ Engine Settings")
    mode = st.radio(
        "Compression Strategy",
        ["⚡ Ultra-Fast Offline (0ms)", "🤖 AI-Powered (Deep Semantic)"],
        index=0,
        help="Fast Offline runs in <1ms without any API calls or internet. AI-Powered uses Gemini for complex prompts."
    )
    
    gemini_key = ""
    if "AI-Powered" in mode:
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Optional if GEMINI_API_KEY environment variable is set."
        )

    st.markdown("---")
    st.subheader("💡 Quick Examples")
    preset = st.selectbox(
        "Load a sample human input:",
        [
            "Select an example...",
            "📄 PDF Scan Request",
            "📊 Sales Data Analysis",
            "💻 Database Query Implementation",
            "🐞 Python Bug Fix",
            "📝 Article Summary"
        ]
    )

    preset_texts = {
        "📄 PDF Scan Request": "do a pdf scan for me",
        "📊 Sales Data Analysis": "Hey! Could you please help me to do an analysis of this sales data and tell me what is inside it? Thanks in advance!",
        "💻 Database Query Implementation": "Hello, I was wondering if you could write some code to connect to PostgreSQL database and fetch users. Thanks!",
        "🐞 Python Bug Fix": "Basically I just want you to figure out why this python script is failing with KeyError",
        "📝 Article Summary": "Can you please kindly do a summary of this article for me asap?"
    }

# Main App Header
st.title("⚡ Prompt Token Optimizer & Analyzer")
st.caption("Convert conversational human requests into token-minimized AI prompts with live token savings.")

# Input text handling
default_input = ""
if preset != "Select an example..." and preset in preset_texts:
    default_input = preset_texts[preset]

user_input = st.text_area(
    "Enter Human Request:",
    value=default_input,
    height=130,
    placeholder="Type naturally (e.g. 'do a pdf scan for me', 'can you please help me write code for...')"
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    optimize_btn = st.button("🚀 Optimize Prompt", type="primary", use_container_width=True)

if user_input:
    orig_tokens = analyzer.count_tokens(user_input)
    st.caption(f"Input Length: **{orig_tokens}** tokens | **{len(user_input)}** characters")

# Execute optimization
if optimize_btn and user_input.strip():
    start_time = time.perf_counter()
    
    optimized_text = ""
    error_msg = None

    if "Ultra-Fast" in mode:
        optimized_text = FastHeuristicCompressor.compress(user_input)
    else:
        ai_compressor = AICompressor(api_key=gemini_key)
        if not ai_compressor.is_available():
            st.warning("⚠️ No Gemini API key provided. Falling back to Ultra-Fast Offline Engine.")
            optimized_text = FastHeuristicCompressor.compress(user_input)
        else:
            with st.spinner("AI is compressing prompt..."):
                optimized_text, error_msg = ai_compressor.compress(user_input)
            if error_msg:
                st.error(f"AI Compression failed: {error_msg}. Using offline fallback.")
                optimized_text = FastHeuristicCompressor.compress(user_input)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Calculate metrics
    metrics = analyzer.analyze_comparison(user_input, optimized_text)

    # Metrics Display Cards
    st.markdown("### 📊 Token Reduction Dashboard")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value-orig">{metrics['original_tokens']}</div>
            <div class="metric-label">Original Tokens</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value-opt">{metrics['optimized_tokens']}</div>
            <div class="metric-label">Optimized Tokens</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value-saved">-{metrics['tokens_saved']} ({metrics['percentage_saved']}%)</div>
            <div class="metric-label">Tokens Saved</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value-saved">{elapsed_ms:.1f} ms</div>
            <div class="metric-label">Processing Time</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Side-by-side prompt view
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("🔴 Raw Human Input")
        st.text_area("Original", value=user_input, height=160, disabled=True, key="orig_disp")

    with c_right:
        st.subheader("🟢 Optimized AI Prompt")
        st.text_area("Optimized", value=optimized_text, height=160, key="opt_disp")
        st.code(optimized_text, language="markdown")
