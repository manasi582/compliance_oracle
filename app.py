import streamlit as st
import time
import os
import sys

# Ensure backend modules can be found
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from backend.rag_pipeline import create_rag_chain

# ========================================================
# UI Configuration
# ========================================================
st.set_page_config(
    page_title="Compliance Oracle",
    page_icon="🤖",
    layout="centered"
)

# Header
st.title("Compliance Oracle")
st.markdown("Ask questions about your company's **compliance policies**.")
st.divider()

# ========================================================
# State Management
# ========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache the heavy resource (the RAG chain)
@st.cache_resource
def get_chain():
    return create_rag_chain()

try:
    chain = get_chain()
except Exception as e:
    st.error(f"Failed to load RAG Pipeline: {e}")
    st.stop()


# ========================================================
# Chat Interface
# ========================================================

# 1. Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle new user input
if prompt := st.chat_input("Ask a policy question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Processing..."):
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Run the RAG pipeline
                    response = chain.invoke(prompt)
                    
                    # Simulate stream typing for effect
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.04)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    break # Success!

                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "rate limit" in error_msg.lower():
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            response = "⚠️ The AI is busy (Rate Limit). Please wait a moment and try again."
                            message_placeholder.error(response)
                    else:
                        response = f" encountered an error: {error_msg}"
                        message_placeholder.error(response)
                        break

        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar options
with st.sidebar:
    st.header("Settings")
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Powered by LangChain & Llama 3")
