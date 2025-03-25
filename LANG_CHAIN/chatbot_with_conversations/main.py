import streamlit as st
import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from config import OPENROUTER_MODELS

# Load environment variables
load_dotenv()

# Constants
MAX_CONVERSATIONS = 3
MAX_MESSAGES = 100
CONV_FILE = "conversations.json"


# Session state initialization
def init_session():
    defaults = {
        "conversations": [],
        "current_conv": None,
        "messages": [],
        "current_model": list(OPENROUTER_MODELS.keys())[0],
        "sidebar_expanded": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# Conversation management
def save_conversations():
    Path(CONV_FILE).write_text(json.dumps(st.session_state.conversations))


def load_conversations():
    try:
        if Path(CONV_FILE).exists():
            conversations = json.loads(Path(CONV_FILE).read_text())
            # Migration for old conversations without model field
            for conv in conversations:
                if "model" not in conv:
                    conv["model"] = list(OPENROUTER_MODELS.keys())[0]
            st.session_state.conversations = conversations
    except Exception as e:
        st.error(f"Error loading conversations: {str(e)}")


def new_conversation():
    conv_id = str(uuid.uuid4())
    new_conv = {
        "id": conv_id,
        "name": f"Chat {len(st.session_state.conversations)+1}",
        "messages": [],
        "model": st.session_state.current_model,
        "timestamp": datetime.now().isoformat(),
    }
    if len(st.session_state.conversations) >= MAX_CONVERSATIONS:
        st.session_state.conversations.pop(0)
    st.session_state.conversations.append(new_conv)
    st.session_state.current_conv = conv_id
    st.session_state.messages = []
    save_conversations()


def delete_conversation(conv_id):
    st.session_state.conversations = [
        c for c in st.session_state.conversations if c["id"] != conv_id
    ]
    if st.session_state.current_conv == conv_id:
        st.session_state.current_conv = None
        st.session_state.messages = []
    save_conversations()


def delete_all_conversations():
    st.session_state.conversations = []
    st.session_state.current_conv = None
    st.session_state.messages = []
    save_conversations()


# Initialize app
init_session()
load_conversations()

# Sidebar with collapsible sections
with st.sidebar:
    # Toggle button
    st.button(
        "☰",
        help="Toggle Sidebar",
        on_click=lambda: st.session_state.update(
            sidebar_expanded=not st.session_state.sidebar_expanded
        ),
    )

    if st.session_state.sidebar_expanded:
        # Model selection
        st.header("Model Configuration")
        prev_model = st.session_state.current_model

        # Get current model from active conversation if exists
        if st.session_state.current_conv:
            current_conv = next(
                c
                for c in st.session_state.conversations
                if c["id"] == st.session_state.current_conv
            )
            st.session_state.current_model = current_conv["model"]

        new_model = st.selectbox(
            "Select Model:",
            options=list(OPENROUTER_MODELS.keys()),
            index=list(OPENROUTER_MODELS.keys()).index(st.session_state.current_model),
        )

        # Handle model change
        if new_model != prev_model:
            # Update current conversation's model if exists
            if st.session_state.current_conv:
                current_conv = next(
                    c
                    for c in st.session_state.conversations
                    if c["id"] == st.session_state.current_conv
                )
                current_conv["model"] = new_model
                save_conversations()

            # Update session state and show system message
            st.session_state.current_model = new_model
            st.session_state.messages.append(
                {"role": "system", "content": f"Model changed to: {new_model}"}
            )
            st.rerun()

        # Conversation management
        st.header("Conversations")
        if st.button("➕ New Chat"):
            new_conversation()
            st.rerun()

        for conv in st.session_state.conversations:
            cols = st.columns([4, 1])
            with cols[0]:
                btn_text = f"{conv['name']} ({conv['model']})"
                if st.button(btn_text, key=f"btn_{conv['id']}"):
                    # Switch to selected conversation
                    st.session_state.current_conv = conv["id"]
                    st.session_state.messages = conv["messages"]
                    st.session_state.current_model = conv["model"]
                    st.rerun()
            with cols[1]:
                if st.button("❌", key=f"del_{conv['id']}"):
                    delete_conversation(conv["id"])
                    st.rerun()

        if st.button("🧹 Delete All", type="primary"):
            delete_all_conversations()
            st.rerun()

# Main chat interface
st.title("AI Chat Assistant")
st.write("Current Model:", st.session_state.current_model)

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "system":
            st.info(msg["content"])
        else:
            st.write(msg["content"])

# Initialize LLM based on current model
model_config = OPENROUTER_MODELS[st.session_state.current_model]
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    model_name=model_config["id"],
    temperature=0,
    streaming=True,
    # model_kwargs={"headers": model_config["headers"]},
)
memory = ConversationBufferMemory(return_messages=True)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
chain = (
    RunnablePassthrough.assign(
        chat_history=lambda _: memory.load_memory_variables({})["history"]
    )
    | prompt
    | llm
)

# Handle user input
if prompt := st.chat_input("Type your message..."):
    # Create new conversation if none exists
    if not st.session_state.current_conv:
        new_conversation()

    current_conv = next(
        c
        for c in st.session_state.conversations
        if c["id"] == st.session_state.current_conv
    )

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    current_conv["messages"].append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        response_placeholder.markdown("▌")

        try:
            for chunk in chain.stream({"input": prompt}):
                if hasattr(chunk, "content"):
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Error: {str(e)}"
            response_placeholder.error(full_response)

    # Update conversation
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    current_conv["messages"].append({"role": "assistant", "content": full_response})
    current_conv["model"] = (
        st.session_state.current_model
    )  # Update model in conversation

    # Enforce limits
    if len(current_conv["messages"]) > MAX_MESSAGES * 2:
        current_conv["messages"] = current_conv["messages"][-MAX_MESSAGES * 2 :]

    current_conv["timestamp"] = datetime.now().isoformat()
    save_conversations()
