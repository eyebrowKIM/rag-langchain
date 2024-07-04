import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer, pipeline

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


class LLM_HuggingFace:
    def __init__(self):
        pass

    def bnb_config(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return bnb_config

    def get_llm(self, model_id="daekeun-ml/phi-2-ko-v0.1"):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=self.bnb_config(), device_map="auto"
        )

        streamer = TextStreamer(tokenizer, skip_prompt=True)

        text_generation_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            device_map="auto",
            top_p=0.9,
            streamer=streamer,
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline, verbose=True)
        return llm
