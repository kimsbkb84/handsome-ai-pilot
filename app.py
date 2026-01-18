import streamlit as st

def main():
    st.set_page_config(page_title="한섬 AI 파일럿", layout="wide")

    st.title("👗 한섬 AI 이미지 태깅 & 검색 시스템")
    st.write("원하시는 작업을 선택해주세요.")
    st.markdown("---")

    # 버튼을 가로로 배치하기 위해 컬럼 사용
    col1, col2 = st.columns(2)

    with col1:
        st.info("새로운 패션 이미지를 등록하고 태깅합니다.")
        # 버튼을 누르면 pages 폴더 안의 upload.py로 이동
        if st.button("1. 이미지 업로드 ➡️", use_container_width=True):
            st.switch_page("pages/upload.py")

    with col2:
        st.info("원하는 스타일의 옷을 검색합니다.")
        # 버튼을 누르면 pages 폴더 안의 search.py로 이동
        if st.button("2. 이미지 검색 ➡️", use_container_width=True):
            st.switch_page("pages/search.py")

if __name__ == "__main__":
    main()