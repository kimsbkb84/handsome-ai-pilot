import os
from dotenv import load_dotenv

# 1. 랭체인과 크로마DB 관련 도구 가져오기
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 2. .env 파일에서 API 키 불러오기
load_dotenv()

# 3. DB가 저장될 폴더 이름 지정
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "handsome_fashion_images"

def get_vector_db():
    # ★ [수정됨] 옛날 모델(embedding-001) 대신 최신 모델(text-embedding-004) 사용
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    vectordb = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )
    
    return vectordb

if __name__ == "__main__":
    db = get_vector_db()
    print("✅ 최신 임베딩 모델(text-embedding-004) 연결 완료!")