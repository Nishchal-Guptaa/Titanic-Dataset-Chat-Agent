# streamlit_app.py

import streamlit as st
import requests
import base64

# --------------------------------------------------
# Configuration
# --------------------------------------------------
API_URL = "https://titanic-dataset-chat-agent-2.onrender.com//chat" 

st.set_page_config(
    page_title="Titanic Chat Agent 🚢",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Dataset Chatbot")
st.markdown(
    "Ask questions about the Titanic dataset and get insights with visualizations."
)

# --------------------------------------------------
# Session State for Chat History
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# Chat Input
# --------------------------------------------------
user_query = st.chat_input("Ask something about the Titanic dataset...")

if user_query:
    # Save user message
    st.session_state.chat_history.append(
        {"role": "user", "content": user_query}
    )

    try:
        # Send request to FastAPI backend
        response = requests.post(
            API_URL,
            json={"query": user_query}
        )

        if response.status_code == 200:
            data = response.json()
            text = data.get("text", "")
            artifact = data.get("artifact", None)

            # Save assistant response
            st.session_state.chat_history.append(
                {"role": "assistant", "content": text, "artifact": artifact}
            )

        else:
            st.error("Error communicating with backend.")

    except Exception as e:
        st.error(f"Connection error: {e}")

# --------------------------------------------------
# Display Chat History
# --------------------------------------------------
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

            # Display image if available
            if message.get("artifact"):
                try:
                    image_bytes = base64.b64decode(message["artifact"])
                    st.image(image_bytes)
                except Exception:
                    st.warning("Could not display visualization.")
