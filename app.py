import streamlit as st
from main import run_orchestrated_pipeline
import pandas as pd
import streamlit.components.v1 as components

st.set_page_config(page_title="Causal Intelligence Agent", layout="centered")
st.title("🧠 Causal Intelligence Agent")

user_input = st.text_area("Paste your text (cause-effect, insights, or anything):", height=200)

def render_mermaid(mermaid_code: str):
    html = f"""
    <div class="mermaid">
        {mermaid_code}
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true }});
    </script>
    """
    components.html(html, height=500, scrolling=True)

if st.button("Run Analysis") and user_input:
    with st.spinner("Analyzing..."):
        result = run_orchestrated_pipeline(user_input)
        task = result["task"]

        if task == "extract":
            st.subheader("📈 Extracted Causal Relationships")
            df = pd.DataFrame(result["relationships"], columns=["Cause", "Effect", "Polarity"])
            st.dataframe(df)

            st.subheader("🌀 Causal Loop Diagram")
            render_mermaid(result["diagram"])

        elif task == "summarize":
            st.subheader("📝 Summary")
            st.markdown(result["summary"])

        else:
            st.subheader("⚠️ Rejected")
            st.markdown(result["message"])
