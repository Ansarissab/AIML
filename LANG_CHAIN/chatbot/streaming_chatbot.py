import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
import os
from dotenv import load_dotenv
from model_config import OPENROUTER_MODELS

load_dotenv()

openai_api_key = os.getenv("OPENROUTER_API_KEY")
openai_api_base = os.getenv("OPENROUTER_BASE_URL")


def initialize_llm(model_config):
    return ChatOpenAI(
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        model_name=model_config["id"],
        temperature=0,
        streaming=True,
        max_retries=2,
    )


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_model" not in st.session_state:
    st.session_state.current_model = list(OPENROUTER_MODELS.values())[0]["id"]
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Sidebar for model selection
with st.sidebar:
    st.header("Model Configuration")
    selected_model_name = st.selectbox(
        "Choose a model:", options=list(OPENROUTER_MODELS.keys()), index=0
    )

    # Get model config
    model_config = OPENROUTER_MODELS[selected_model_name]
    selected_model_id = model_config["id"]

    # Handle model change
    if selected_model_id != st.session_state.current_model:
        st.session_state.messages.append(
            {"role": "system", "content": f"Model changed to: {selected_model_name}"}
        )
        st.session_state.current_model = selected_model_id
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
        st.rerun()

# Initialize components with current model
llm = initialize_llm(OPENROUTER_MODELS[selected_model_name])
memory = st.session_state.memory

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Be polite."),
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

# Main chat interface
st.title("LangChain + OpenRouter Multi-Model Chat Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "system":
            st.info(msg["content"])
        else:
            st.write(msg["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        response_placeholder.markdown("▌")

        try:
            for chunk in chain.stream({"input": user_input}):
                if hasattr(chunk, "content"):
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            full_response = "Sorry, I couldn't process that request. Please try again."
            response_placeholder.markdown(full_response)

    # Update state and memory
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(full_response)
