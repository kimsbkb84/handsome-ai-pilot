"""
한섬 AI 파일럿 - 이미지 벡터 검색 조회 화면 (search.py)

- 자연어 검색 / 태그 기반 검색 분리
- 유사도 점수 시각화 (Progress bar)
- 카드형 결과 UI, 정렬 옵션
- 추천·최근 검색어 (session_state)
※ upload·vectordb 저장 로직은 변경하지 않음.
"""

import streamlit as st
import os
import re
from enum import Enum
from typing import Any

from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from vectordb import get_vector_db

# -----------------------------------------------------------------------------
# [설정] 상수 및 기본값 (하드코딩 최소화)
# -----------------------------------------------------------------------------

# 검색 결과 개수 범위
DEFAULT_TOP_K = 10
MIN_TOP_K = 1
MAX_TOP_K = 20

# 벡터 DB 거리(distance) → 유사도 퍼센트 변환용
# Chroma는 L2 거리 반환: 작을수록 유사. 이 값으로 0~100% 스케일링.
SCORE_DISTANCE_MAX = 2.0  # 거리 >= 이 값이면 0% 근처로 처리
SCORE_DISTANCE_MIN = 0.0  # 거리 0 = 100%

# 세션 스테이트 키
SK_RECENT_SEARCHES = "search_recent_searches"
SK_LAST_QUERY = "search_last_query"
SK_LAST_RESULTS = "search_last_results"
MAX_RECENT_SEARCHES = 10

# 추천 검색어 (DB에서 태그를 못 가져올 때만 사용, 확장 가능)
FALLBACK_SUGGESTIONS = [
    "원피스, 네이비, 오피스룩",
    "여름에 입기 좋은 흰색 원피스",
    "린넨, 캐주얼, 미니멀",
    "코트, 트위드, 오버사이즈",
]


class SortOption(str, Enum):
    """검색 결과 정렬 기준 (추가 확장 시 여기만 수정)."""
    BY_SIMILARITY = "similarity"      # 유사도 높은 순 (기본)
    BY_LATEST = "latest"              # 최신 업로드 순 (메타데이터에 날짜 있을 때)
    BY_TAG_MATCH = "tag_match"        # 태그 일치 개수 순


# -----------------------------------------------------------------------------
# [초기화] 환경변수 및 페이지 설정
# -----------------------------------------------------------------------------

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("API 키가 없습니다. .env 파일을 확인해주세요.")
    st.stop()

# Streamlit 베스트 프랙티스: 페이지 설정은 최상단 한 번만
st.set_page_config(page_title="이미지 검색 | 한섬 AI 파일럿", layout="wide")


# -----------------------------------------------------------------------------
# [세션 상태] 최근 검색어 초기화
# -----------------------------------------------------------------------------

def _init_session_state():
    """최근 검색어 등 검색 관련 session_state 초기화."""
    if SK_RECENT_SEARCHES not in st.session_state:
        st.session_state[SK_RECENT_SEARCHES] = []
    if SK_LAST_QUERY not in st.session_state:
        st.session_state[SK_LAST_QUERY] = ""
    if SK_LAST_RESULTS not in st.session_state:
        st.session_state[SK_LAST_RESULTS] = []


def _push_recent_search(query: str) -> None:
    """검색 실행 시 최근 검색어 리스트에 추가 (중복·빈 문자열 제외, 최대 개수 유지)."""
    q = (query or "").strip()
    if not q:
        return
    recent = st.session_state.get(SK_RECENT_SEARCHES, [])
    if q in recent:
        recent.remove(q)
    recent.insert(0, q)
    st.session_state[SK_RECENT_SEARCHES] = recent[:MAX_RECENT_SEARCHES]


_init_session_state()


# -----------------------------------------------------------------------------
# [유틸] 거리 → 유사도 퍼센트
# -----------------------------------------------------------------------------

def distance_to_confidence_percent(distance: float) -> float:
    """
    벡터 DB에서 반환된 거리(distance)를 0~100% 유사도로 변환.
    Chroma L2: 작을수록 유사. SCORE_DISTANCE_MAX 기준으로 선형 스케일.
    """
    if distance <= SCORE_DISTANCE_MIN:
        return 100.0
    if distance >= SCORE_DISTANCE_MAX:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - distance / SCORE_DISTANCE_MAX)))


# -----------------------------------------------------------------------------
# [데이터] DB에서 사용 가능한 태그 목록 추출
# -----------------------------------------------------------------------------

def get_available_tags() -> list[str]:
    """
    vectorDB 메타데이터/문서에서 태그 후보를 동적으로 추출.
    DB 접근 실패 또는 문서 없으면 FALLBACK_SUGGESTIONS 기반 태그 후보 반환.
    """
    try:
        db = get_vector_db()
        # LangChain Chroma: _collection.get()으로 전체 문서/메타데이터 조회
        coll = getattr(db, "_collection", None)
        if coll is None:
            return _tags_from_fallback()
        raw = coll.get(include=["documents", "metadatas"])
        documents = raw.get("documents") or []
        tags_set = set()
        for doc_list in documents:
            for doc in doc_list if isinstance(doc_list, list) else [doc_list]:
                if not isinstance(doc, str):
                    continue
                # 저장 형식: "원피스, 네이비, 롱기장, 린넨, ..."
                for part in re.split(r"[,，\s]+", doc.strip()):
                    t = part.strip()
                    if len(t) >= 1 and len(t) <= 30:
                        tags_set.add(t)
        return sorted(tags_set) if tags_set else _tags_from_fallback()
    except Exception:
        return _tags_from_fallback()


def _tags_from_fallback() -> list[str]:
    """FALLBACK_SUGGESTIONS에서 단일 태그/키워드만 추출해 반환 (중복 제거)."""
    tags_set = set()
    for s in FALLBACK_SUGGESTIONS:
        for part in re.split(r"[,，\s]+", s.strip()):
            t = part.strip()
            if t:
                tags_set.add(t)
    return sorted(tags_set)


# -----------------------------------------------------------------------------
# [검색] 벡터 검색 실행 (점수 포함)
# -----------------------------------------------------------------------------

def run_vector_search(query_text: str, k: int) -> list[tuple[Any, float]]:
    """
    자연어 또는 태그 조합 문자열로 벡터 검색 수행.
    반환: [(Document, distance), ...] (distance는 작을수록 유사).
    """
    if not (query_text or query_text.strip()):
        return []
    try:
        db = get_vector_db()
        # similarity_search_with_score: (Document, distance) 리스트
        pairs = db.similarity_search_with_score(query_text.strip(), k=k)
        return list(pairs)
    except Exception:
        return []


# -----------------------------------------------------------------------------
# [정렬] 검색 결과 정렬 (SortOption enum 사용)
# -----------------------------------------------------------------------------

def sort_search_results(
    pairs: list[tuple[Any, float]],
    option: SortOption,
) -> list[tuple[Any, float]]:
    """
    (Document, distance) 리스트를 선택한 정렬 기준에 따라 정렬.
    - BY_SIMILARITY: 거리 오름차순 (이미 DB 기본 순서일 수 있음)
    - BY_LATEST: 메타데이터 uploaded_at 내림차순 (키 없으면 유사도 유지)
    - BY_TAG_MATCH: page_content(태그 문자열) 길이 내림차순으로 대리 (일치 개수 근사)
    """
    if not pairs:
        return pairs

    if option == SortOption.BY_SIMILARITY:
        return sorted(pairs, key=lambda x: x[1])

    if option == SortOption.BY_LATEST:
        def key_latest(item):
            doc, dist = item
            meta = doc.metadata or {}
            # 업로드 날짜 메타데이터 확장 시 여기만 수정
            uploaded = meta.get("uploaded_at") or meta.get("created_at") or ""
            return (uploaded, -item[1])  # 날짜 없으면 유사도로 2차 정렬
        return sorted(pairs, key=key_latest, reverse=True)

    if option == SortOption.BY_TAG_MATCH:
        # 태그 일치 개수 대신 문서 태그 개수(길이)로 정렬 (확장 시 쿼리 태그와 매칭 개수로 교체 가능)
        def key_tag_match(item):
            doc, dist = item
            content = (doc.page_content or "").strip()
            n = len([x for x in re.split(r"[,，\s]+", content) if x.strip()])
            return (n, -dist)
        return sorted(pairs, key=key_tag_match, reverse=True)

    return pairs


# -----------------------------------------------------------------------------
# [UI 컴포넌트] 결과 카드 한 개 (재사용)
# -----------------------------------------------------------------------------

def render_result_card(
    doc: Any,
    rank: int,
    confidence_percent: float,
) -> None:
    """
    검색 결과 한 건을 카드 형태로 렌더링.
    - 순위, 유사도(Progress bar), 메타: source, image_type, 태그(chip), 카테고리/날짜/시즌/브랜드(옵션).
    - 이미지 URL/경로는 현재 DB에 없으므로 플레이스홀더; 추후 메타데이터 확장 시 연동.
    """
    meta = doc.metadata or {}
    source = meta.get("source", "(파일명 없음)")
    image_type = meta.get("image_type", "")
    # 업로드 날짜·카테고리·시즌·브랜드: upload에서 저장 시 추가 가능하도록 키만 사용
    uploaded_at = meta.get("uploaded_at") or "—"
    category = meta.get("category") or meta.get("image_type") or "—"
    season = meta.get("season") or "—"
    brand = meta.get("brand") or "—"

    tags_text = (doc.page_content or "").strip()
    tag_list = [t.strip() for t in re.split(r"[,，]+", tags_text) if t.strip()]

    with st.container():
        st.markdown(f"**{rank}. {source}**")
        # 유사도 Progress bar
        st.caption(f"유사도 **{confidence_percent:.0f}%**")
        st.progress(confidence_percent / 100.0)
        # 태그 칩 (가로 배치, 최대 8개)
        if tag_list:
            show_tags = tag_list[:8]
            chip_cols = st.columns(len(show_tags))
            for idx, tag in enumerate(show_tags):
                with chip_cols[idx]:
                    st.markdown(f"`{tag}`")
        else:
            st.caption("태그 없음")
        # 메타 정보 한 줄
        st.caption(f"카테고리: {category} · 업로드: {uploaded_at} · 시즌: {season} · 브랜드: {brand}")
        st.markdown("---")


# -----------------------------------------------------------------------------
# [UI] 추천 검색어 / 최근 검색어 블록
# -----------------------------------------------------------------------------

def render_suggestions_and_recent():
    """
    입력창 하단에 추천 검색어 + 최근 검색어 노출.
    추천: get_available_tags() 또는 FALLBACK 기반; 최근: session_state.
    """
    st.caption("💡 추천 검색어")
    suggestions = get_available_tags()
    if suggestions:
        # 태그가 많으면 상위 N개만 (슬라이스로 확장 가능)
        display_tags = suggestions[:12]
        for i in range(0, len(display_tags), 4):
            row = st.columns(4)
            for j, col in enumerate(row):
                idx = i + j
                if idx < len(display_tags):
                    with col:
                        if st.button(
                            display_tags[idx],
                            key=f"suggest_{idx}_{display_tags[idx][:20]}",
                            use_container_width=True,
                        ):
                            st.session_state[SK_LAST_QUERY] = display_tags[idx]
                            st.rerun()
    st.caption("🕒 최근 검색어")
    recent = st.session_state.get(SK_RECENT_SEARCHES, [])
    if recent:
        for i, q in enumerate(recent[:5]):
            if st.button(q, key=f"recent_{i}_{hash(q) % 10**6}", use_container_width=True):
                st.session_state[SK_LAST_QUERY] = q
                st.rerun()
    else:
        st.caption("_아직 최근 검색어가 없습니다._")


# -----------------------------------------------------------------------------
# [메인] 검색 방식 분리: 자연어 / 태그 기반
# -----------------------------------------------------------------------------

st.title("🔍 패션 이미지 태그 검색 (Pilot)")
st.caption("자연어 또는 태그로 저장된 이미지 정보를 검색합니다.")

# 검색 방식 탭
tab_natural, tab_tags = st.tabs(["🔍 자연어 검색", "🏷 태그 기반 검색"])

# ----- 자연어 검색 -----
with tab_natural:
    st.caption("예: \"여름에 입기 좋은 흰색 원피스\"")
    natural_query = st.text_input(
        "검색어를 입력하세요",
        value=st.session_state.get(SK_LAST_QUERY, ""),
        placeholder="예: 여름에 입기 좋은 흰색 원피스",
        key="natural_search_input",
    )
    render_suggestions_and_recent()
    btn_natural = st.button("🔍 자연어 검색", type="primary", key="btn_natural")

# ----- 태그 기반 검색 -----
with tab_tags:
    st.caption("태그를 선택하면 조합해서 검색합니다.")
    available_tags = get_available_tags()
    selected_tags = st.multiselect(
        "태그 선택 (복수 가능)",
        options=available_tags,
        default=[],
        key="tag_multiselect",
    )
    tag_query = ", ".join(selected_tags) if selected_tags else ""
    btn_tag = st.button("🏷 태그로 검색", type="primary", key="btn_tag")

# 공통: 결과 개수, 정렬 옵션
col1, col2 = st.columns([1, 2])
with col1:
    top_k = st.slider("가져올 결과 개수", min_value=MIN_TOP_K, max_value=MAX_TOP_K, value=DEFAULT_TOP_K)
with col2:
    sort_option = st.selectbox(
        "정렬",
        options=[SortOption.BY_SIMILARITY, SortOption.BY_LATEST, SortOption.BY_TAG_MATCH],
        format_func=lambda x: {
            SortOption.BY_SIMILARITY: "🔥 유사도 높은 순",
            SortOption.BY_LATEST: "🕒 최신 업로드 순",
            SortOption.BY_TAG_MATCH: "🏷 태그 일치 개수 순",
        }[x],
        index=0,
        key="sort_option",
    )

# 검색 실행: 어느 탭에서 버튼을 눌렀는지에 따라 사용할 쿼리 결정
search_triggered = False
active_query = ""
if btn_natural and (natural_query or "").strip():
    active_query = natural_query.strip()
    search_triggered = True
elif btn_tag and tag_query:
    active_query = tag_query
    search_triggered = True

# ----- 검색 실행 및 결과 렌더링 -----
if search_triggered and active_query:
    _push_recent_search(active_query)
    st.session_state[SK_LAST_QUERY] = active_query

    with st.spinner("검색 중..."):
        try:
            pairs = run_vector_search(active_query, k=top_k)
            pairs = sort_search_results(pairs, sort_option)

            st.divider()
            st.subheader("📋 검색 결과")

            if not pairs:
                st.info("조건에 맞는 결과가 없습니다.")
            else:
                st.success(f"총 **{len(pairs)}건** 찾았습니다.")
                for i, (doc, distance) in enumerate(pairs, 1):
                    confidence = distance_to_confidence_percent(distance)
                    render_result_card(doc, i, confidence)
        except Exception as e:
            st.error(f"검색 중 오류가 발생했습니다: {e}")
