# config.py
from os import getenv
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_base = getenv("OPENROUTER_BASE_URL")

OPENROUTER_MODELS = {
    "Mistral 7B Instruct": {
        "id": "mistralai/mistral-7b-instruct:free",
        "headers": {
            "HTTP-Referer": openai_api_base,
            "X-Title": "AI Chat App",
        },
    },
    "Mixtral 8x7B Instruct": {
        "id": "mistralai/mixtral-8x7b-instruct:free",
        "headers": {
            "HTTP-Referer": openai_api_base,
            "X-Title": "AI Chat App",
        },
    },
    "Google Gemma 32B": {
        "id": "google/gemma-3-27b-it:free",
        "headers": {
            "HTTP-Referer": openai_api_base,
            "X-Title": "AI Chat App",
            "Content-Type": "application/json",
        },
    },
    "Llama 2 13B Chat": {
        "id": "meta-llama/llama-2-13b-chat:free",
        "headers": {
            "HTTP-Referer": openai_api_base,
            "X-Title": "AI Chat App",
        },
    },
    "NVIDIA": {
        "id": "nvidia/llama-3.1-nemotron-70b-instruct:free",
        "headers": {
            "HTTP-Referer": openai_api_base,
            "X-Title": "AI Chat App",
        },
    },
    "Deepseek R1 Distill Qwen 32b": {
        "id": "deepseek/deepseek-r1-distill-qwen-32b:free",
        "headers": {
            "HTTP-Referer": openai_api_base,
            "X-Title": "AI Chat App",
        },
    },
    "Deepseek V3": {
        "id": "deepseek/deepseek-chat-v3-0324:free",
        "headers": {
            "HTTP-Referer": openai_api_base,
            "X-Title": "AI Chat App",
        },
    },
    "Deepseek R1": {
        "id": "deepseek/deepseek-r1:free",
        "headers": {
            "HTTP-Referer": openai_api_base,
            "X-Title": "AI Chat App",
        },
    },
    "Microsoft phi 3": {
        "id": "microsoft/phi-3-medium-128k-instruct:free",
        "headers": {
            "HTTP-Referer": openai_api_base,
            "X-Title": "AI Chat App",
        },
    },
    "Dolphin R1": {
        "id": "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
        "headers": {
            "HTTP-Referer": openai_api_base,
            "X-Title": "AI Chat App",
        },
    },
}
