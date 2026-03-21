# LLM path
model_path = {
    "qwen_7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_72b": "Qwen/Qwen2.5-72B-Instruct",
    "llama_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "llama_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def resolve_llm_model_path(llm_model_name, llm_model_path=""):
    if llm_model_path and llm_model_path.strip():
        return llm_model_path
    return model_path[llm_model_name]
