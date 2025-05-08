import os
from dotenv import load_dotenv, find_dotenv
import torch

_ = load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class ChatModelFactory:
    model_params = {
        "temperature": 0,
        "seed": 42,
    }

    @classmethod
    def get_model(cls, model_name: str, use_azure: bool = False):
        if "gpt" in model_name:
            if not use_azure:
                return ChatOpenAI(model=model_name, **cls.model_params)
            else:
                return AzureChatOpenAI(
                    azure_deployment=model_name,
                    api_version="2024-05-01-preview",
                    **cls.model_params
                )
        elif model_name == "deepseek":
            # 使用硅基流动的 API
            return ChatOpenAI(
                model="deepseek-ai/DeepSeek-V3",
                base_url="https://api.siliconflow.cn/v1",
                api_key=os.getenv("SILICONFLOW_API_KEY"),
                temperature=0.1,
                max_tokens=512
            )
        elif model_name == "qwen":
            # 使用 Qwen 模型
            model_id = "Qwen/Qwen1.5-1.8B-Chat"  # 使用 1.8B 版本
            model_path = os.path.join(os.getenv("MODEL_PATH", "./models"), "Qwen1.5-1.8B-Chat")
            
            # 确保模型目录存在
            os.makedirs(model_path, exist_ok=True)
            
            # 抑制 tokenizer 的警告
            import warnings
            warnings.filterwarnings("ignore", message="Special tokens have been added in the vocabulary")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=model_path,
                local_files_only=False,  # 如果本地没有模型，则下载
                trust_remote_code=True,
                padding_side="left",  # 添加填充方向
                truncation=True  # 启用截断
            )
            
            # 强制使用 float32 并禁用混合精度
            torch_dtype = torch.float32
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=model_path,
                local_files_only=False,  # 如果本地没有模型，则下载
                device_map="auto",  # 自动分配设备
                torch_dtype=torch_dtype,  # 强制使用 float32
                trust_remote_code=True,
                offload_folder=os.path.join(model_path, "offload"),  # 添加模型卸载目录
                offload_state_dict=True,  # 启用状态字典卸载
                low_cpu_mem_usage=True  # 启用低内存模式
            )
            
            # 二次强制转换并禁用自动混合精度
            model = model.to(torch.float32)
            model.eval()
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                device_map="auto",  # 使用自动设备映射
                pad_token_id=tokenizer.pad_token_id,  # 添加填充标记
                eos_token_id=tokenizer.eos_token_id,  # 添加结束标记
                torch_dtype=torch_dtype  # 强制使用 float32
            )
            return ChatHuggingFace(
                llm=HuggingFacePipeline(pipeline=pipe),
                model_id=model_id,
                model_kwargs={"temperature": 0.1, "max_new_tokens": 512}
            )
        elif model_name == "ollama":
            # 使用Ollama的API
            return ChatOllama(
                base_url=os.getenv("OLLAMA_BASE_URL"),
                model="qwen2.5:72b",
                temperature=0,
                num_ctx=16384,
                num_predict=-1,
            )
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")

    @classmethod
    def get_default_model(cls):
        return cls.get_model("qwen")  # 修改默认模型为 qwen


class EmbeddingModelFactory:

    @classmethod
    def get_model(cls, model_name: str, use_azure: bool = False):
        if model_name.startswith("text-embedding"):
            if not use_azure:
                return OpenAIEmbeddings(model=model_name)
            else:
                return AzureOpenAIEmbeddings(
                    azure_deployment=model_name,
                    openai_api_version="2024-05-01-preview",
                )
        elif model_name == "BAAI":
            # 使用HuggingFace的本地嵌入模型
            model_path = os.path.join(os.getenv("MODEL_PATH", "./models"), "paraphrase-multilingual-MiniLM-L12-v2")
            
            # 确保模型目录存在
            os.makedirs(model_path, exist_ok=True)
            
            # 根据 CUDA 可用性选择设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 使用更小的多语言模型
                cache_folder=model_path,  # 指定模型缓存路径
                model_kwargs={"device": device},  # 使用确定的设备
                encode_kwargs={"normalize_embeddings": True}
            )
        elif model_name == "MINI":
            # 使用更小的嵌入模型
            model_path = os.path.join(os.getenv("MODEL_PATH", "./models"), "all-MiniLM-L6-v2")
            
            # 确保模型目录存在
            os.makedirs(model_path, exist_ok=True)
            
            # 根据 CUDA 可用性选择设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # 使用更小的模型
                cache_folder=model_path,  # 指定模型缓存路径
                model_kwargs={"device": device},  # 使用确定的设备
                encode_kwargs={"normalize_embeddings": True}
            )
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")

    @classmethod
    def get_default_model(cls):
        return cls.get_model("MINI")  # 修改默认模型为 MINI
