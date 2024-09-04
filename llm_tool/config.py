import os


MODEL_NAME_MAPPING = {
    "palm2": "models/text-bison-001",
    "gemini-pro": "models/gemini-pro",
    "gpt-3.5-1106": "gpt-3.5-turbo-1106",
    "Llama-2-7b-chat": "Llama-2-7b-chat",
    "Llama-2-13b-chat": "Llama-2-13b-chat",
    "Llama-2-70b-chat": "Llama-2-70b-chat",
}
BASE_MODEL_LST = ["palm2", "gemini-pro"]
CHAT_MODEL_LST = ["gpt-3.5-1106", "Llama-2-7b-chat", "Llama-2-13b-chat", "Llama-2-70b-chat"]

# TODO: Add your API key here, if you have multiple keys, you can add them to the list
# We implemented a random selection mechanism to avoid hitting the rate limit
GOOGLE_API_KEYS = [
    os.getenv("PALM_API_KEY"),
    os.getenv("PALM_API_KEY_2"),
    os.getenv("PALM_API_KEY_3"),
]

OPENAI_API_KEY = [
    os.getenv("OPENAI_API_KEY_NLG")
]

AZURE_API_KEY = {
    "Llama-2-7b-chat": os.getenv("AZURE_LLAMA_7B_CHAT"),
    "Llama-2-13b-chat": os.getenv("AZURE_LLAMA_13B_CHAT"),
    "Llama-2-70b-chat": os.getenv("AZURE_LLAMA_70B_CHAT"),
}

def get_azure_endpoint(model_name):
    if model_name in BASE_MODEL_LST:
        return f"https://{model_name}-kw-serverless.eastus2.inference.ai.azure.com/v1/completions"
    elif model_name in CHAT_MODEL_LST:
        return f"https://{model_name}-kw-serverless.eastus2.inference.ai.azure.com/v1/chat/completions"
    else:
        raise ValueError(f"Invalid model name: {model_name}")
