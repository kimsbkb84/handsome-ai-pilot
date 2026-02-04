import streamlit as st
import os
import time
import uuid  # 유일한 파일명 생성용
import json  # ★ JSON 처리를 위해 추가
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from vectordb import get_vector_db
from brand_data import get_brand_from_filename # 브랜드 장부

# 1. 환경변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("API 키가 없습니다. .env 파일을 확인해주세요.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# [기능 1] 구글 드라이브 업로드 (Placeholder)
# ==========================================
def upload_to_google_drive(file_obj, filename):
    # 지금은 테스트용 가짜 링크 반환
    return f"[https://fake-drive-link.com/](https://fake-drive-link.com/){filename}"

# ==========================================
# [기능 2] 태깅 생성 (JSON 최적화 프롬프트)
# ==========================================
def generate_tags(image):
    model = genai.GenerativeModel('gemini-flash-lite-latest')
    
    # ★ 비용 절감 & 정교함을 위한 영어/JSON 프롬프트
    prompt = """
    Role: You are a Senior Merchandiser (MD) at Handsome with 20 years of experience.
    Task: Analyze the visual elements of the image and extract structured data for search optimization.

    [Constraints & Rules]
    1. **Output Format**: Return ONLY a valid JSON object. No markdown.
    2. **Language**: Values must be in Korean.
    3. **Detail-Oriented**: Focus on specific design elements (buttons, neckline, fit).

    [Controlled Vocabulary]
    - Season: [SS, FW, All_Season]
    - Style: [Minimal, Casual, Feminine, Classic, Street, Formal]
    - Fit: [Slim, Regular, Loose, Oversized, Cropped]

    [JSON Structure]
    {
      "cat": "Detailed Item Name (e.g., 크롭 트위드 재킷)",
      "col": ["Main Color", "Sub Color"],
      "mat": "Visual Texture (e.g., 트위드, 부클, 실크)",
      "pat": "Pattern (e.g., 솔리드, 체크, 하운드투스)",
      "sty": "Style Keyword",
      "sea": "Season",
      "neck": "Neckline Type (e.g., 라운드넥, V넥, 카라, 후드)",
      "fit": "Fit Type",
      "det": ["Detail 1", "Detail 2", "Detail 3"] 
    }
    
    [Example Output]
    {
      "cat": "노카라 트위드 재킷", 
      "col": ["아이보리", "골드"], 
      "mat": "트위드", 
      "pat": "솔리드", 
      "sty": "Feminine", 
      "sea": "SS",
      "neck": "라운드넥",
      "fit": "Cropped",
      "det": ["금장 단추", "프린지 마감", "포켓 디테일", "배색 라인"]
    }
    """
    
    try:
        response = model.generate_content([prompt, image])
        text = response.text.strip()
        
        # [안전장치] AI가 ```json ... ``` 형태로 줄 경우 앞뒤 제거
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        return text.strip()
        
    except Exception as e:
        # 에러 발생 시 JSON 형식의 에러 메시지 반환
        return json.dumps({"error": str(e)})

# ==========================================
# [UI] 화면 구성
# ==========================================
st.title("☁️ 한섬 AI 포토 클라우드 (Pilot)")
st.caption("이미지 자동 식별(UUID) + JSON 태깅 + 구글 드라이브 연동")

uploaded_files = st.file_uploader(
    "이미지를 드래그하세요 (파일명 예: TM_코트.jpg)", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) > 10:
    st.warning("⚠️ 파일럿 테스트는 최대 10장까지만 가능합니다.")
    st.stop()

col1, col2 = st.columns([1, 1])
with col1:
    start_btn = st.button("🚀 업로드 및 처리 시작", type="primary", use_container_width=True)
with col2:
    cancel_btn = st.button("🔄 초기화", use_container_width=True)

if cancel_btn:
    st.rerun()

# ==========================================
# [핵심 로직] 실행
# ==========================================
if start_btn and uploaded_files:
    
    total_files = len(uploaded_files)
    st.divider()
    
    with st.status(f"⚙️ 데이터 처리 중... (총 {total_files}장)", expanded=True) as status:
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            current_idx = i + 1
            
            # 1. 브랜드 코드 추출
            original_name = uploaded_file.name
            brand_name = get_brand_from_filename(original_name)
            
            status.write(f"**[{current_idx}/{total_files}]** {original_name} (브랜드: {brand_name})")
            
            # 2. UUID 생성
            file_ext = os.path.splitext(original_name)[1]
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            
            # 3. 구글 드라이브 업로드 (가짜 링크)
            drive_link = upload_to_google_drive(uploaded_file, unique_filename)
            
            # 4. 이미지 태깅 (JSON 문자열 받기)
            image = Image.open(uploaded_file)
            json_str = generate_tags(image)
            
            # 5. [핵심] JSON 파싱 및 DB 저장
            try:
                # 5-1. JSON 문자열을 딕셔너리로 변환
                data = json.loads(json_str)
                
                # 에러 체크 (AI가 에러를 뱉었을 경우)
                if "error" in data:
                    raise Exception(data["error"])

                # 5-2. 검색용 텍스트 만들기
                # 리스트 형태인 색상(['네이비', '화이트'])을 문자열("네이비 화이트")로 변환
                colors = " ".join(data.get('col', [])) if isinstance(data.get('col'), list) else str(data.get('col'))
                
                # 검색할 때 걸리게 할 단어들을 조합 (줄글 형태)
                search_text = f"{data.get('cat')} {colors} {data.get('sty')} {data.get('mat')} {data.get('pat')} {data.get('sea')}"
                
                # 5-3. DB 저장
                db = get_vector_db()
                db.add_texts(
                    texts=[search_text], # 임베딩(검색)은 이 줄글로 하고
                    metadatas=[{
                        "original_name": original_name,
                        "uuid_name": unique_filename,
                        "brand": brand_name,
                        "drive_link": drive_link,
                        "image_type": "fashion",
                        "detail_json": json_str  # ★ 원본 JSON도 통째로 저장 (나중에 상세화면에 씀)
                    }]
                )

                if data.get('neck') in ["없음", "None", "해당없음"]:
                    data['neck'] = ""
                
                if data.get('fit') in ["Regular"] and "의류" not in data.get('cat', ''):
                    # 옷이 아닌데 Regular 핏이라고 하면 지워버림
                    data['fit'] = ""

                # 검색용 텍스트 만들기 (청소된 데이터로 다시 조합)
                # 빈 값은 자동으로 빠지게 됨
                search_text = f"{data.get('cat')} {colors} {data.get('sty')} {data.get('mat')} {data.get('neck')} {data.get('fit')} {data.get('det')}"
                
                # 성공 로그
                status.write(f"   └ ✅ 태깅/저장 완료: {data.get('cat')} / {data.get('sty')}")
                
            except json.JSONDecodeError:
                st.error(f"❌ JSON 파싱 실패 ({uploaded_file.name}): AI 응답 형식이 올바르지 않습니다.")
                st.code(json_str) # 디버깅용으로 뭘 뱉었는지 보여줌
            except Exception as e:
                st.error(f"❌ 처리 중 오류 발생: {e}")
            
            progress_bar.progress(current_idx / total_files)
            
            # 무료 티어 속도 조절
            if i < total_files - 1:
                time.sleep(15) 
        
        status.update(label="🎉 모든 작업이 완료되었습니다!", state="complete", expanded=False)

    st.success("작업 완료! DB에 JSON 메타데이터가 잘 들어갔습니다.")
    
    # (선택) 결과 미리보기
    with st.expander("👀 마지막 데이터 확인 (상세 보기)", expanded=True):
        
        # 1:1.5 비율로 왼쪽(이미지)과 오른쪽(정보)을 나눔
        col_img, col_info = st.columns([1, 1.5])
        
        # [왼쪽] 이미지 표시
        with col_img:
            st.image(image, caption="최종 처리된 이미지", use_container_width=True)
            
        # [오른쪽] 텍스트 정보 표시
        with col_info:
            st.markdown("### 📄 파일 정보")
            st.markdown(f"**원본 파일명:** `{original_name}`")
            st.markdown(f"**UUID 키값:** `{unique_filename}`")
            st.markdown(f"**브랜드:** `{brand_name}`")
            
            st.divider()
            
            st.markdown("### 🧩 AI 태깅 결과 (JSON)")
            # JSON 문자열을 예쁜 딕셔너리 형태로 보여줌
            try:
                st.json(json.loads(json_str))
            except:
                # 만약 파싱에 실패했다면 원본 텍스트로 보여줌
                st.warning("JSON 파싱에 실패하여 텍스트로 표시합니다.")
                st.code(json_str, language="json")

elif not uploaded_files and start_btn:
    st.warning("이미지를 먼저 선택해주세요.")