import os

import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms.openllm import OpenLLM
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


class LLMFactory:
    def __init__(self, model_choice, model, openai_api_key=None, temperature=0, streaming=True):
        self.model_choice = model_choice
        self.model = model
        self.openai_api_key = openai_api_key
        self.temperature = temperature
        self.streaming = streaming

        self.llm_instance = self.create_llm_instance()

    def create_llm_instance(self):
        if self.model_choice == "Local":
            return LLM_HuggingFace(self.model)
        elif self.model_choice == "Docker":
            return LLM_Docker(self.model)
        elif self.model_choice == "OpenAI":
            return LLM_OpenAI(self.model, self.openai_api_key, self.temperature, self.streaming)
        else:
            raise ValueError("Unsupported model choice")


class LLM_HuggingFace:
    def __init__(self, model_name):
        self.model_name = model_name
        self.config = self.bnb_config()
        self.setup_environment()

    @staticmethod
    def bnb_config():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    def setup_environment(self):
        os.environ["TRANSFORMERS_CACHE"] = "./cache/"
        os.environ["HF_HOME"] = "./cache/"

    def get_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            # quantization_config=self.config,
            device_map="auto",
            cache_dir="./cache/model/",
            trust_remote_code=True,
        )
        return model

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name, cache_dir="./cache/", trust_remote_code=True)

    def get_pipeline(self):
        model = self.get_model()
        tokenizer = self.get_tokenizer()
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=4096)
        return pipe

    def get_llm(self):
        return HuggingFacePipeline(pipeline=self.get_pipeline())


class CustomCallbackHandler(StreamingStdOutCallbackHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_llm_new_token(self, token: str):
        # Implement here your streaming logic
        print(token, end="", flush=True)


class LLM_Docker:
    def __init__(self, server_url):
        self.server_url = server_url
        self.callback = CustomCallbackHandler()

    def get_llm(self):
        return OpenLLM(server_url=self.server_url, callbacks=[self.callback], timeout=100)


class LLM_OpenAI:
    def __init__(self, model_name, openai_api_key, temperature=0, streaming=True):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.temperature = temperature
        self.streaming = streaming

    def get_llm(self):
        return ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            streaming=self.streaming,
        )


if __name__ == "__main__":
    # 예제 사용
    llm_factory = LLMFactory("OpenAI", "gpt-3", openai_api_key="your_api_key", temperature=0.5)
    llm_instance = llm_factory.create_llm_instance()
