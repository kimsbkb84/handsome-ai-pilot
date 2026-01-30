import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from vectordb import get_vector_db

# 1. 환경변수(.env)에서 API 키 불러오기
# 보안을 위해 코드에 키를 직접 적지 않고 .env에서 가져옵니다.
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 2. 구글 Gemini API 설정
# 키가 없으면 에러를 띄웁니다.
if not GOOGLE_API_KEY:
    st.error("API 키가 없습니다. .env 파일을 확인해주세요.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# [핵심 함수] 구글 Gemini에게 이미지 태깅 요청하기
# ==========================================
def generate_tags(image):
    """
    이미지를 받아 Gemini 1.5 Flash 모델에게 태깅을 요청하는 함수입니다.
    """
    # 1) 모델 선택: Gemini 1.5 Flash (속도가 빠르고 무료 티어 사용 가능)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    # 2) [프롬프트 엔지니어링] 패션 전문가 페르소나 부여
    # AI에게 명확한 역할과 출력 형식을 지정해줍니다.
    prompt = """
    너는 한섬(Handsome)의 패션 전문 MD야. 
    이 이미지를 분석해서 쇼핑몰 검색에 도움이 될 핵심 태그를 추출해줘.
    
    [분석 항목]
    1. 카테고리 (예: 원피스, 셔츠, 코트)
    2. 색상 (예: 네이비, 아이보리, 파스텔톤)
    3. 소재 느낌 (예: 실크, 데님, 니트, 트위드)
    4. 패턴 (예: 스트라이프, 플로럴, 무지)
    5. 스타일/무드 (예: 캐주얼, 오피스룩, 미니멀, 빈티지)
    
    [출력 형식]
    반드시 아래와 같이 '키워드 나열' 형태로만 대답해. 불필요한 문장은 쓰지 마.
    예시: 원피스, 네이비, 롱기장, 린넨, 여름, 오피스룩, 반팔
    """
    
    # 3) API 호출 (이미지와 프롬프트를 함께 전송)
    try:
        response = model.generate_content([prompt, image])
        return response.text.strip() # 결과 텍스트 반환
    except Exception as e:
        return f"에러 발생: {str(e)}"

# ==========================================
# [UI 구성] 화면 디자인 시작
# ==========================================
st.title("📸 패션 이미지 AI 태깅 (Pilot)")
st.caption("이미지를 업로드하면 AI가 자동으로 태그를 생성합니다. (최대 10장)")

# 1. 파일 업로더 (여러 파일 드래그 앤 드롭 가능)
uploaded_files = st.file_uploader(
    "이미지를 드래그해서 놓으세요", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)

# 2. [제한 조건] 파일 개수 10개 제한 로직
if len(uploaded_files) > 4:
    st.warning(f"⚠️ 파일럿 테스트를 위해 최대 4장까지만 가능합니다. (현재 {len(uploaded_files)}장)")
    # 10장이 넘으면 이후 로직을 실행하지 않고 멈춤
    st.stop()

# 3. 버튼 영역 (컬럼으로 나누어 배치)
col1, col2 = st.columns([1, 1])

with col1:
    start_btn = st.button("🚀 태깅 시작", type="primary", use_container_width=True)

with col2:
    # 취소 버튼은 사실상 새로고침 역할을 합니다.
    cancel_btn = st.button("🔄 초기화 (취소)", use_container_width=True)

if cancel_btn:
    st.rerun() # 화면 새로고침

# ==========================================
# [실행 로직] 태깅 시작 버튼을 눌렀을 때
# ==========================================
if start_btn and uploaded_files:
    
    st.divider() # 구분선
    st.subheader("📝 분석 결과")
    
    # 진행 상황을 보여주는 바(Bar) 생성
    progress_bar = st.progress(0)
    
    # 업로드된 파일들을 하나씩 꺼내서 처리
    for i, uploaded_file in enumerate(uploaded_files):
        
        # 1) 이미지를 PIL 형식으로 변환 (Streamlit -> PIL)
        image = Image.open(uploaded_file)
        
        # 2) 화면에 이미지와 로딩 상태 표시 (2분할)
        result_col1, result_col2 = st.columns([1, 2])
        
        with result_col1:
            st.image(image, use_container_width=True, caption=uploaded_file.name)
            
        with result_col2:
            with st.spinner(f"'{uploaded_file.name}' 분석 중..."):
                # 1. 태깅 생성
                tags = generate_tags(image)
                
                # 2. 결과 출력
                st.success("분석 완료!")
                st.markdown(f"**🏷️ 생성된 태그:**")
                st.info(tags)
                
                # ==========================================
                # ★ [추가된 부분] 크로마DB에 저장하기
                # ==========================================
                try:
                    db = get_vector_db() # 창고 문 열기
                    
                    # DB에 데이터 넣기 (텍스트: 태그, 메타데이터: 파일이름)
                    db.add_texts(
                        texts=[tags], 
                        metadatas=[{"source": uploaded_file.name, "image_type": "fashion"}]
                    )
                    st.toast(f"💾 DB 저장 완료: {uploaded_file.name}") # 저장 알림창 띄우기
                    
                except Exception as e:
                    st.error(f"DB 저장 실패: {e}")
                # ==========================================
        
        # 진행률 업데이트
        progress_bar.progress((i + 1) / len(uploaded_files))
        
    st.success("모든 이미지의 태깅 작업이 완료되었습니다!")

# 파일은 올렸는데 버튼을 안 눌렀을 때 안내 문구
elif not uploaded_files and start_btn:
    st.warning("먼저 이미지를 업로드해주세요.")