import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Set
import hashlib
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.chains import RetrievalQA

from Models.Factory import ChatModelFactory, EmbeddingModelFactory

class FileLoadFactory:
    @staticmethod
    def get_loader(filename: str):
        ext = get_file_extension(filename)
        if ext == "pdf":
            return PyMuPDFLoader(filename)
        elif ext == "docx" or ext == "doc":
            return UnstructuredWordDocumentLoader(filename)
        else:
            raise NotImplementedError(f"File extension {ext} not supported.")

def get_file_extension(filename: str) -> str:
    return filename.split(".")[-1]

def load_docs(filename: str) -> List[Document]:
    file_loader = FileLoadFactory.get_loader(filename)
    pages = file_loader.load_and_split()
    return pages

def get_file_hash(filename: str) -> str:
    """计算文件的哈希值"""
    with open(filename, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def normalize_path(path: str) -> str:
    """统一路径格式"""
    # 转换为相对于工作目录的路径
    rel_path = os.path.relpath(path, os.getcwd())
    # 统一使用正斜杠
    return rel_path.replace('\\', '/')

def get_persist_path(filename: str) -> str:
    """根据文件名生成唯一的持久化目录路径"""
    # 使用相对于工作目录的路径
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    persist_dir = os.path.join("data", "chroma_dbs", name)
    print(f"持久化目录路径: {persist_dir}")
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir

def load_document_metadata(persist_dir: str) -> dict:
    """加载文档元数据"""
    metadata_file = os.path.join(persist_dir, 'document_metadata.json')
    print(f"尝试加载元数据文件: {metadata_file}")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                print(f"成功加载元数据: {metadata}")
                return metadata
        except Exception as e:
            print(f"加载元数据失败: {str(e)}")
    return {"documents": {}, "last_update": ""}

def save_document_metadata(persist_dir: str, metadata: dict):
    """保存文档元数据"""
    metadata_file = os.path.join(persist_dir, 'document_metadata.json')
    print(f"保存元数据到: {metadata_file}")
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print("元数据保存成功")
    except Exception as e:
        print(f"保存元数据失败: {str(e)}")

def check_document_changes(filename: str, persist_dir: str) -> tuple[bool, Set[str]]:
    """检查文档变化并返回需要更新的文档ID集合"""
    current_hash = get_file_hash(filename)
    metadata = load_document_metadata(persist_dir)
    
    # 使用统一格式的相对路径
    normalized_path = normalize_path(filename)
    print(f"检查文档: {normalized_path}, 当前哈希: {current_hash}")
    
    # 清理元数据中的重复路径
    cleaned_docs = {}
    for path in metadata["documents"]:
        cleaned_path = normalize_path(path)
        cleaned_docs[cleaned_path] = metadata["documents"][path]
    metadata["documents"] = cleaned_docs
    
    # 如果文档不存在于元数据中，或者哈希值发生变化
    if normalized_path not in metadata["documents"]:
        print(f"文档 {normalized_path} 不在元数据中，需要添加")
        metadata["documents"][normalized_path] = current_hash
        metadata["last_update"] = current_hash
        save_document_metadata(persist_dir, metadata)
        return True, set()
    elif metadata["documents"][normalized_path] != current_hash:
        print(f"文档 {normalized_path} 哈希值发生变化: {metadata['documents'][normalized_path]} -> {current_hash}")
        metadata["documents"][normalized_path] = current_hash
        metadata["last_update"] = current_hash
        save_document_metadata(persist_dir, metadata)
        return True, set()
    else:
        print(f"文档 {normalized_path} 未发生变化")
        return False, set(metadata["documents"].keys())

def check_database_exists(persist_dir: str) -> bool:
    """检查数据库文件是否存在"""
    # 检查 SQLite 数据库文件
    db_file = os.path.join(persist_dir, 'chroma.sqlite3')
    if os.path.exists(db_file):
        return True
    
    # 检查是否有 UUID 命名的目录
    for item in os.listdir(persist_dir):
        if os.path.isdir(os.path.join(persist_dir, item)) and len(item.split('-')) == 5:
            return True
    
    return False

def ask_document_persistent(
        filename: str,
        query: str,
        model_name: str = "deepseek",  # 默认使用 deepseek 模型
        embedding_name: str = "MINI",  # 默认使用 MINI 嵌入模型
) -> str:
    """基于持久化向量数据库的文档问答"""
    print("开始处理文档...")
    persist_dir = get_persist_path(filename)
    print(f"使用持久化目录: {persist_dir}")
    
    # 检查数据库文件是否存在
    db_exists = check_database_exists(persist_dir)
    print(f"数据库文件存在: {db_exists}")
    
    embedding_model = EmbeddingModelFactory.get_model(embedding_name)
    print("已加载嵌入模型")
    
    # 检查文档变化
    has_changes, existing_docs = check_document_changes(filename, persist_dir)
    print(f"文档变化检查结果 - 有变化: {has_changes}, 现有文档: {existing_docs}")
    
    # 检查持久化目录下是否已有向量数据库
    if db_exists and not has_changes:
        print("找到现有的向量数据库，正在加载...")
        try:
            db = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
            print("数据库加载成功")
        except Exception as e:
            print(f"加载数据库失败: {str(e)}")
            db = None
    else:
        if has_changes:
            print("检测到文档内容变化，更新向量数据库...")
        else:
            print("未找到向量数据库，开始处理文档...")
        
        # 加载现有数据库（如果存在）
        if db_exists:
            try:
                db = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
                print("成功加载现有数据库")
            except Exception as e:
                print(f"加载现有数据库失败: {str(e)}")
                db = None
        else:
            db = None
        
        # 处理新文档
        raw_docs = load_docs(filename)
        print(f"已加载文档，页数: {len(raw_docs)}")
        
        if len(raw_docs) == 0:
            return "抱歉，文档内容为空"
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        print("开始分割文档...")
        documents = text_splitter.split_documents(raw_docs)
        print(f"文档分割完成，共 {len(documents)} 个片段")
        
        if documents is None or len(documents) == 0:
            return "无法读取文档内容"
        
        # 如果是首次创建数据库
        if db is None:
            print("开始创建向量数据库...")
            try:
                db = Chroma.from_documents(documents, embedding_model, persist_directory=persist_dir)
                print("数据库创建成功")
            except Exception as e:
                print(f"创建数据库失败: {str(e)}")
                return f"创建数据库失败: {str(e)}"
        else:
            print("开始更新向量数据库...")
            try:
                # 添加新文档到现有数据库
                db.add_documents(documents)
                print("数据库更新成功")
            except Exception as e:
                print(f"更新数据库失败: {str(e)}")
                return f"更新数据库失败: {str(e)}"
    
    print("开始构建问答链...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatModelFactory.get_model(model_name),
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True,  # 返回源文档
        verbose=True  # 显示详细处理过程
    )
    print("开始查询...")
    result = qa_chain.invoke({"query": query + "(请用中文回答)"})  # 使用 invoke 而不是 run
    print("查询完成")
    
    # 清理输出文本
    response = result["result"]
    # 移除特殊标记
    response = response.replace("<|im_start|>", "").replace("<|im_end|>", "")
    # 移除重复的文本
    response = "\n".join(dict.fromkeys(response.split("\n")))
    
    return response

if __name__ == "__main__":
    filename = "data/供应商资格要求.pdf"
    print(f"File path: {filename}")
    query = "总结供应商资格要求pdf"
    response = ask_document_persistent(filename, query)
    print(response) 