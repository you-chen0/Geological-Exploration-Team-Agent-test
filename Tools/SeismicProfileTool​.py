from typing import List, Optional
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from langchain.tools import StructuredTool
from langchain.vectorstores import Chroma
from langchain.schema import Document

class SeismicProfileRAG:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", persist_dir: str = "./seismic_profiles_db"):
        """
        初始化地震剖面RAG系统
        Args:
            model_name: 使用的图像编码模型名称
            persist_dir: 向量数据库持久化目录
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载图像处理器和模型
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # 初始化向量数据库
        self.vectorstore = None
        
    def _encode_image(self, image_path: str) -> List[float]:
        """将图像编码为向量"""
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        return outputs[0].detach().numpy().tolist()
    
    def add_images(self, image_paths: List[str], descriptions: Optional[List[str]] = None):
        """
        添加地震剖面图像到数据库
        Args:
            image_paths: 图像文件路径列表
            descriptions: 对应的描述列表（可选）
        """
        documents = []
        for i, image_path in enumerate(image_paths):
            # 编码图像
            embedding = self._encode_image(image_path)
            
            # 创建文档
            metadata = {
                "image_path": image_path,
                "type": "seismic_profile"
            }
            if descriptions and i < len(descriptions):
                metadata["description"] = descriptions[i]
            
            doc = Document(
                page_content=f"Seismic profile image: {Path(image_path).name}",
                metadata=metadata
            )
            documents.append(doc)
            
        # 存储到向量数据库
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=lambda x: self._encode_image(x.metadata["image_path"]),
                persist_directory=str(self.persist_dir)
            )
        else:
            self.vectorstore.add_documents(documents)
        self.vectorstore.persist()
    
    def search_similar_profiles(self, query_image_path: str, k: int = 3) -> List[dict]:
        """
        搜索相似的地震剖面图像
        Args:
            query_image_path: 查询图像路径
            k: 返回结果数量
        Returns:
            相似图像的信息列表
        """
        if self.vectorstore is None:
            return []
            
        # 编码查询图像
        query_embedding = self._encode_image(query_image_path)
        
        # 搜索相似图像
        results = self.vectorstore.similarity_search_by_vector(
            query_embedding,
            k=k
        )
        
        # 格式化结果
        similar_profiles = []
        for doc in results:
            profile_info = {
                "image_path": doc.metadata["image_path"],
                "similarity_score": doc.metadata.get("score", 0),
                "description": doc.metadata.get("description", "No description available")
            }
            similar_profiles.append(profile_info)
            
        return similar_profiles

def search_seismic_profiles(query_image: str, k: int = 3) -> str:
    """搜索相似的地震剖面图像"""
    rag = SeismicProfileRAG()
    results = rag.search_similar_profiles(query_image, k)
    
    if not results:
        return "未找到相似的地震剖面图像。"
        
    response = "找到以下相似的地震剖面图像：\n\n"
    for i, result in enumerate(results, 1):
        response += f"{i}. 图像路径: {result['image_path']}\n"
        response += f"   相似度得分: {result['similarity_score']:.4f}\n"
        response += f"   描述: {result['description']}\n\n"
    
    return response
