import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rag_chain import build_chain
from sentiment import classify, SentimentResult

load_dotenv()

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "My calls keep dropping — what should I do?",
    "How do I activate international roaming?",
    "Why is my bill higher than usual this month?",
    "My phone shows SIM not detected after a restart",
    "How do I enable Wi-Fi calling?",
    "I was charged for roaming but had a bundle active",
    "How do I unlock my phone for another network?",
]

PRIORITY_COLORS = {
    "low":    "#4CAF50",
    "medium": "#FF9800",
    "high":   "#F44336",
}

st.set_page_config(
    page_title="Telecom Support Chat",
    page_icon="📡",
    layout="centered",
)

@st.cache_resource
def get_chain():
    return build_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "sentiments" not in st.session_state:
    st.session_state.sentiments = {}

with st.sidebar:
    st.title("📡 Telecom Support")
    st.caption("Powered by RAG · Qwen3-32B on Groq")
    st.divider()

    if st.session_state.sentiments:
        st.markdown("**📊 Session Sentiment**")
        results = list(st.session_state.sentiments.values())
        latest: SentimentResult = results[-1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mood", f"{latest.emoji} {latest.sentiment.capitalize()}")
        with col2:
            st.metric("Priority", latest.priority.upper())

        high_count = sum(1 for r in results if r.priority == "high")
        if high_count:
            st.warning(f"⚠️ {high_count} high-priority message(s) this session")

        st.divider()

    st.markdown("**Sample questions**")
    st.caption("Click one to send it instantly.")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sentiments = {}

st.title("Customer Care Assistant")
st.caption("Ask me anything about your mobile service — connectivity, billing, SIM, roaming, and more.")

user_msg_index = 0
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])

            if user_msg_index in st.session_state.sentiments:
                s: SentimentResult = st.session_state.sentiments[user_msg_index]
                badge_color = PRIORITY_COLORS.get(s.priority, "gray")
                st.markdown(
                    f'<span style="font-size:0.75rem; color:{badge_color}; '
                    f'border:1px solid {badge_color}; border-radius:4px; '
                    f'padding:1px 6px; margin-right:4px;">'
                    f'{s.priority.upper()}</span>'
                    f'<span style="font-size:0.75rem; color:gray;">'
                    f'{s.emoji} {s.sentiment}</span>',
                    unsafe_allow_html=True,
                )
            user_msg_index += 1
        else:
            st.markdown(msg["content"])


question = st.chat_input("Describe your issue…")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    current_user_index = user_msg_index

    with st.spinner("Analysing…"):
        sentiment: SentimentResult = classify(question)
    st.session_state.sentiments[current_user_index] = sentiment

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
        badge_color = PRIORITY_COLORS.get(sentiment.priority, "gray")
        st.markdown(
            f'<span style="font-size:0.75rem; color:{badge_color}; '
            f'border:1px solid {badge_color}; border-radius:4px; '
            f'padding:1px 6px; margin-right:4px;">'
            f'{sentiment.priority.upper()}</span>'
            f'<span style="font-size:0.75rem; color:gray;">'
            f'{sentiment.emoji} {sentiment.sentiment}</span>',
            unsafe_allow_html=True,
        )

    if sentiment.escalate:
        st.warning(
            "🚨 **We've detected high frustration.** "
            "Would you like us to escalate this to a human agent? "
            "Call **611** or use the **MyTelecom app** → *Contact Us*.",
            icon="📞",
        )

    # Build chat history from session state (exclude the message just appended)
    chat_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        chain = get_chain()
        response = st.write_stream(
            chain.stream({"question": question, "chat_history": chat_history})
        )

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
