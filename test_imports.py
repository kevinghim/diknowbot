import streamlit as st

st.title("Testing Imports")

try:
    import anthropic
    st.success("✅ anthropic package is installed")
except ImportError as e:
    st.error(f"❌ Error importing anthropic: {e}")

try:
    from langchain_anthropic import ChatAnthropic
    st.success("✅ langchain_anthropic package is installed")
except ImportError as e:
    st.error(f"❌ Error importing langchain_anthropic: {e}") 