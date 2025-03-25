import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from os import getenv
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API credentials and model name
openai_api_key = getenv("OPENROUTER_API_KEY")
openai_api_base = getenv("OPENROUTER_BASE_URL")
model_name = getenv("OPENAI_MODEL_NAME")

# Initialize the OpenRouter LLM model with streaming enabled
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    model_name=model_name,
    temperature=0,
    streaming=True,
)

# Define a chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Be Polite."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Set up conversational memory
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Create a ConversationChain with the LLM and memory
conversation = ConversationChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

# Streamlit app layout
st.title("LangChain, OpenRouter ChatGPT-like Chatbot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if user_input := st.chat_input("Ask me anything:"):
    # Display user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate and display bot's response with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Show initial thinking indicator
        response_placeholder.markdown("▌")

        # Stream the response
        for chunk in conversation.stream({"input": user_input}):
            # Extract content from the response chunk
            full_response += chunk.get("response", "")
            response_placeholder.markdown(full_response + "▌")

        # Remove cursor after completion
        response_placeholder.markdown(full_response)

    # Add final response to messages
    st.session_state.messages.append({"role": "assistant", "content": full_response})
