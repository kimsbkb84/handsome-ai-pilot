# check_db.py
from vectordb import get_vector_db
import pandas as pd

def view_chroma_data():
    print("🔍 크로마DB 데이터 조회 중...")
    
    # 1. DB 연결
    db = get_vector_db()
    
    # 2. 모든 데이터 가져오기
    # ChromaDB에서 데이터를 꺼낼 때는 .get()을 씁니다.
    data = db.get()
    
    # 3. 데이터 개수 확인
    count = len(data['ids'])
    print(f"\n📊 현재 저장된 데이터 개수: {count}개")
    
    if count == 0:
        print("텅 비어있습니다! upload.py에서 저장을 먼저 해주세요.")
        return

    # 4. 보기 좋게 출력 (판다스 표 활용)
    print("\n[최신 데이터 5개 미리보기]")
    df = pd.DataFrame({
        'ID': data['ids'],
        '태그내용(Embeddings)': data['documents'], # 여기에 태그가 들어있습니다
        '파일출처(Metadata)': data['metadatas']
    })
    
    # 표 출력
    print(df.tail(5)) # 가장 최근에 들어간 5개 출력

if __name__ == "__main__":
    view_chroma_data()