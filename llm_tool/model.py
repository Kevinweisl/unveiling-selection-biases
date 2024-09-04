import random
import os
import sys
import json
import requests
from datetime import datetime

from openai import OpenAI
from google.generativeai.types import safety_types
import google.generativeai as GoogleAI

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from .utils import retry_with_exponential_backoff, AzureRateLimitError, AzureServerError
from config import MODEL_NAME_MAPPING, \
    GOOGLE_API_KEYS, OPENAI_API_KEY, AZURE_API_KEY, \
    BASE_MODEL_LST, CHAT_MODEL_LST, \
    get_azure_endpoint


class LLMModel:
    def __init__(self, **kwargs):
        self.model_display_name = kwargs.get("model_name")
        self.model_name = MODEL_NAME_MAPPING.get(self.model_display_name)

        if self.model_display_name is None:
            raise ValueError("model_name must be specified")
        elif self.model_display_name in BASE_MODEL_LST:
            self.type = "base"
        elif self.model_display_name in CHAT_MODEL_LST:
            self.type = "chat"
        else:
            raise ValueError("model_name must be a valid model name")

    def generate(self, **kwargs):
        if kwargs.get("prompt") is None:
            raise ValueError("prompt must be specified")

        if self.type == "base":
            return self.generate_base(**kwargs)
        elif self.type == "chat":
            return self.generate_chat(**kwargs)

    def generate_base(self, **kwargs):
        raise NotImplementedError

    def generate_chat(self, **kwargs):
        raise NotImplementedError

    def get_client(self):
        raise NotImplementedError

    def process_result(self, **kwargs):
        raise NotImplementedError


class OpenAIModel(LLMModel):
    # Text Completion Docs: https://platform.openai.com/docs/api-reference/completions/create
    # Chat Completion Docs: https://platform.openai.com/docs/api-reference/chat/create
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @retry_with_exponential_backoff
    def generate_base(self, **kwargs):
        client = self.get_client()

        kwargs["completion"] = client.completions.create(
            model=self.model_name,
            prompt=kwargs.get("prompt", "How are you?"),
            temperature=kwargs.get("temperature", 0),
            logprobs=kwargs.get("logprobs", 5),
            n=kwargs.get("candidate_count", 1),
        )

        return self.process_result(**kwargs)

    @retry_with_exponential_backoff
    def generate_chat(self, **kwargs):
        client = self.get_client()

        kwargs["completion"] = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": kwargs.get("prompt", "How are you?")}
            ],
            temperature=kwargs.get("temperature", 0),
            logprobs=True,
            top_logprobs=kwargs.get("logprobs", 5),
            n=kwargs.get("candidate_count", 1),
        )

        return self.process_result(**kwargs)

    def get_client(self):
        self.api_key = random.choice(OPENAI_API_KEY)
        client = OpenAI(api_key=self.api_key)

        return client

    def process_result(self, **kwargs):
        completion = kwargs.get("completion")

        if self.type == "base":
            return {
                "model_name": completion.model,
                "model_display_name": self.model_display_name,
                "masked_api_key": self.api_key[-5:],
                "result": completion.choices[0].text,
                "candidates": [choice.text for choice in completion.choices],
                "info": {
                    'id': completion.id,
                    'object': completion.object,
                    'created': completion.created,
                    'usage': {
                        'completion_tokens': completion.usage.completion_tokens,
                        'prompt_tokens': completion.usage.prompt_tokens,
                        'total_tokens': completion.usage.total_tokens
                    },
                    'choices': [
                        json.loads(choice.model_dump_json())
                        for choice in completion.choices
                    ]
                }
            }
        elif self.type == "chat":
            return {
                "model_name": completion.model,
                "model_display_name": self.model_display_name,
                "masked_api_key": self.api_key[-5:],
                "result": completion.choices[0].message.content,
                "candidates": [choice.message.content for choice in completion.choices],
                "info": {
                    'id': completion.id,
                    'object': completion.object,
                    'created': completion.created,
                    'system_fingerprint': completion.system_fingerprint,
                    'usage': {
                        'completion_tokens': completion.usage.completion_tokens,
                        'prompt_tokens': completion.usage.prompt_tokens,
                        'total_tokens': completion.usage.total_tokens
                    },
                    'choices': [
                        json.loads(choice.model_dump_json())
                        for choice in completion.choices
                    ]
                }
            }


class GoogleAIModel(LLMModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_random_seed = datetime.now().timestamp()

    def get_client(self):
        random.seed(self.model_random_seed + datetime.now().timestamp())
        self.api_key = random.choice(GOOGLE_API_KEYS)
        GoogleAI.configure(api_key=self.api_key)

        return GoogleAI


class PaLM2Model(GoogleAIModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @retry_with_exponential_backoff
    def generate_base(self, **kwargs):
        client = self.get_client()

        kwargs["completion"] = client.generate_text(
            model=self.model_name,
            prompt=kwargs.get("prompt", "How are you?"),
            temperature=kwargs.get("temperature", 0),
            candidate_count=kwargs.get("candidate_count", 5),
            max_output_tokens=kwargs.get("max_output_tokens", 1024),
            safety_settings=[
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_SEXUAL,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
            ]
        )

        return self.process_result(**kwargs)

    @retry_with_exponential_backoff
    def generate_chat(self, **kwargs):
        pass

    def process_result(self, **kwargs):
        completion = kwargs.get("completion")
        ignore_safety_ratings = kwargs.get("ignore_safety_ratings", True)

        if self.type == "base":
            return {
                "model_name": self.model_name,
                "model_display_name": self.model_display_name,
                "masked_api_key": self.api_key[-5:],
                "result": completion.result,
                "candidates": [
                    candidate["output"]
                    for candidate in completion.candidates
                ] if ignore_safety_ratings else completion.candidates,
            }
        elif self.type == "chat":
            raise NotImplementedError


class GeminiModel(GoogleAIModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @retry_with_exponential_backoff
    def generate_base(self, **kwargs):
        client = self.get_client()
        model = client.GenerativeModel(model_name=self.model_display_name)

        kwargs["completion"] = model.generate_content(
            kwargs.get("prompt", "How are you?"),
            generation_config={
                'temperature': kwargs.get("temperature", 0),
                'max_output_tokens': kwargs.get("max_output_tokens", 1024),
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]
        )

        return self.process_result(**kwargs)

    def process_result(self, **kwargs):
        completion = kwargs.get("completion")
        ignore_safety_ratings = kwargs.get("ignore_safety_ratings", True)

        if self.type == "base":
            try:
                result = completion.text
            except Exception as e:
                if str(completion.prompt_feedback) == "block_reason: OTHER\n":
                    result = "[[block_reason: OTHER]]"
                else:
                    raise e
            return {
                "model_name": self.model_name,
                "model_display_name": self.model_display_name,
                "masked_api_key": self.api_key[-5:],
                "result": result,
                "candidates":  [
                    candidate.content.parts[0].text
                    for candidate in completion.candidates
                ] if ignore_safety_ratings else completion.candidates,
            }
        elif self.type == "chat":
            raise NotImplementedError


class LlamaModel(LLMModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.url = get_azure_endpoint(self.model_display_name)
        self.api_key = AZURE_API_KEY[self.model_display_name]
        self.headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ self.api_key)}

    @retry_with_exponential_backoff
    def generate_base(self, **kwargs):
        data =  {
            "prompt": kwargs.get("prompt", "How are you?"),
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_output_tokens", 1024),
            "n": kwargs.get("candidate_count", 1),
            "logprobs": kwargs.get("logprobs", 5)
        }

        re = requests.post(self.url, headers=self.headers, data=json.dumps(data))
        kwargs["completion"] = json.loads(re.text)

        return self.process_result(**kwargs)

    @retry_with_exponential_backoff
    def generate_chat(self, **kwargs):
        data =  {
            "messages": [
                {
                "role": "user",
                "content": kwargs.get("prompt", "How are you?"),
                }
            ],
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_output_tokens", 1024),
            "n": kwargs.get("candidate_count", 1),
            "logprobs": kwargs.get("logprobs", 5) # not supported yet
        }

        re = requests.post(self.url, headers=self.headers, data=json.dumps(data))
        kwargs["completion"] = json.loads(re.text)

        if re.status_code == 429 and kwargs["completion"]["message"].startswith("Rate Limit"):
            raise AzureRateLimitError
        elif re.status_code == 500:
            raise AzureServerError

        return self.process_result(**kwargs)

    def process_result(self, **kwargs):
        completion = kwargs.get("completion")

        response = {
            "model_name": self.model_name,
            "model_display_name": self.model_display_name,
            "masked_api_key": self.api_key[-5:],
        }

        if self.is_blocked_content(completion):
            response["result"] = "[[block_reason: OTHER]]"
            response["candidates"] = []

            return response

        response["info"] = {
            'id': completion.get("id"),
            'object': completion.get("object"),
            'created': completion.get("created"),
            'usage': completion.get("usage"),
            'choices': [
                choice
                for choice in completion.get("choices")
            ]
        }

        if self.type == "base":
            response["result"] = completion["choices"][0]["text"]
            response["candidates"] = [
                candidate["text"]
                for candidate in completion["choices"]
            ]
            return response
        elif self.type == "chat":
            response["result"] = completion["choices"][0]["message"]["content"]
            response["candidates"] = [
                candidate["message"]["content"]
                for candidate in completion["choices"]
            ]
            return response

    def is_blocked_content(self, completion):
        if completion.get('error') and completion['error'].get('code') == 'content_filter':
            return True
        elif completion.get('choices') and completion['choices'][0].get('finish_reason') == 'content_filter':
            return True
        else:
            return False
