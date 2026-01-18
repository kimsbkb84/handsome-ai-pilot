import os
from dotenv import load_dotenv

# 1. 랭체인과 크로마DB 관련 도구 가져오기
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 2. .env 파일에서 API 키 불러오기 (이 코드가 없으면 에러남)
load_dotenv()

# 3. DB가 저장될 폴더 이름 지정 (내 컴퓨터에 이 폴더가 생김)
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "handsome_fashion_images"

def get_vector_db():
    """
    벡터 DB를 불러오거나, 없으면 새로 만드는 함수입니다.
    """
    # 임베딩 모델 설정 (이미지/텍스트를 숫자로 바꾸는 기계)
    # 비용 절약을 위해 가장 저렴한 'text-embedding-3-small' 모델 사용
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # ChromaDB 로드 (없으면 자동 생성)
    vectordb = Chroma(
        persist_directory=CHROMA_PERSIST_DIR, # 이 폴더에 데이터 저장
        embedding_function=embedding_model,   # 이 기계로 벡터화
        collection_name=COLLECTION_NAME       # 데이터베이스 이름
    )
    
    print(f"📂 벡터 DB 로드 완료: {CHROMA_PERSIST_DIR}")
    return vectordb

# 테스트용 코드 (이 파일을 직접 실행했을 때만 작동)
if __name__ == "__main__":
    db = get_vector_db()
    print("✅ DB 세팅 성공! 이제 데이터를 넣을 준비가 되었습니다.")