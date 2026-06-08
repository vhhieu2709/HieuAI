import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
import joblib
import unicodedata
from PIL import Image

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="ASL → Tiếng Việt",
    page_icon="🤟",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

.stApp { background:#0a0a0f; color:#e0e0e0; font-family:'Inter',sans-serif; }

.main-title {
    font-family:'Space Mono',monospace; font-size:2.4rem; font-weight:700;
    background:linear-gradient(135deg,#00ff88,#00cfff);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    text-align:center; margin-bottom:0.2rem;
}
.sub-title { text-align:center; color:#666; font-size:0.9rem; margin-bottom:2rem; font-family:'Space Mono',monospace; }

.result-box {
    background:#13131a; border:1px solid #1e1e2e; border-radius:16px;
    padding:1.5rem; margin:0.5rem 0; text-align:center;
}
.result-label { font-size:0.72rem; color:#555; text-transform:uppercase; letter-spacing:2px; margin-bottom:0.5rem; font-family:'Space Mono',monospace; }
.result-value { font-size:3.5rem; font-weight:700; font-family:'Space Mono',monospace; color:#00ff88; }
.result-value-viet { font-size:1.8rem; font-weight:600; font-family:'Inter',sans-serif; color:#00cfff; word-break:break-all; min-height:2.5rem; }

.conf-bar-wrap { background:#1e1e2e; border-radius:8px; height:8px; margin-top:0.8rem; overflow:hidden; }
.conf-bar { height:100%; border-radius:8px; background:linear-gradient(90deg,#00ff88,#00cfff); }

.top3-row { display:flex; justify-content:space-between; margin-top:0.8rem; }
.top3-item { background:#1a1a28; border-radius:10px; padding:0.5rem 0.8rem; text-align:center; flex:1; margin:0 3px; }
.top3-letter { font-size:1.4rem; font-weight:700; color:#fff; font-family:'Space Mono',monospace; }
.top3-conf { font-size:0.72rem; color:#888; }

.guide-box {
    background:#13131a; border-left:3px solid #00ff88;
    border-radius:0 12px 12px 0; padding:0.9rem 1.1rem;
    margin:0.4rem 0; font-size:0.83rem; color:#aaa; font-family:'Space Mono',monospace;
    line-height:1.7;
}

.stButton>button {
    background:linear-gradient(135deg,#00ff88,#00cfff) !important;
    color:#0a0a0f !important; border:none !important; border-radius:10px !important;
    font-weight:700 !important; font-family:'Space Mono',monospace !important;
    padding:0.55rem 1rem !important; width:100% !important;
}
.stButton>button:hover { opacity:0.82 !important; }

hr { border-color:#1e1e2e !important; }
#MainMenu,footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL (cache)
# ============================================================
@st.cache_resource
def load_model():
    bundle = joblib.load("model_baseline.pkl")
    return bundle["model"], bundle["label_encoder"]

@st.cache_resource
def load_mediapipe():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    )


# ============================================================
# TRÍCH XUẤT LANDMARKS
# ============================================================
def extract_landmarks(img_pil: Image.Image, hands) -> np.ndarray | None:
    img_rgb = np.array(img_pil.convert("RGB"))
    result  = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None

    lm     = result.multi_hand_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm.landmark])  # (21,3)

    # Chuẩn hóa: dịch về wrist, scale theo max abs
    coords -= coords[0]
    scale   = np.max(np.abs(coords)) + 1e-6
    coords /= scale
    return coords.flatten()   # (63,)


# ============================================================
# TELEX DECODER
# ============================================================
TELEX_VOWELS = {"aa":"â","ee":"ê","oo":"ô","ow":"ơ","uw":"ư","aw":"ă","dd":"đ"}
TELEX_TONES  = {"s":"\u0301","f":"\u0300","r":"\u0309","x":"\u0303","j":"\u0323"}

def telex_decode(s: str) -> str:
    s = s.lower()
    for t, v in TELEX_VOWELS.items():
        s = s.replace(t, v)
    words = []
    for word in s.split(" "):
        if len(word) >= 2 and word[-1] in TELEX_TONES:
            tone, base = TELEX_TONES[word[-1]], word[:-1]
            for i in range(len(base)-1, -1, -1):
                if base[i] in "aăâeêioôơuưy":
                    word = base[:i] + unicodedata.normalize("NFC", base[i]+tone) + base[i+1:]
                    break
            else:
                word = base
        words.append(word)
    return " ".join(words)


# ============================================================
# SESSION STATE
# ============================================================
for k, v in [("buffer", []), ("output_words", []), ("last_pred", None), ("last_conf", 0.0), ("last_top3", [])]:
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# LOAD RESOURCES
# ============================================================
model, le      = load_model()
hands_detector = load_mediapipe()
CLASS_NAMES    = list(le.classes_)   # ['A','B',...,'Space','Z']


# ============================================================
# GIAO DIỆN
# ============================================================
st.markdown('<div class="main-title">🤟 ASL → Tiếng Việt</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">MediaPipe · XGBoost · Accuracy 99.3% · 27 ký hiệu ASL</div>', unsafe_allow_html=True)

col_cam, col_result = st.columns([1.2, 1], gap="large")

# ──── CỘT TRÁI: Camera / Upload ────────────────────────────
with col_cam:
    st.markdown("#### 📷 Nguồn ảnh")
    source = st.radio("", ["📷 Camera", "🖼️ Upload ảnh"], horizontal=True, label_visibility="collapsed")

    img_input = None
    if source == "📷 Camera":
        cam = st.camera_input("Giơ tay lên và chụp ảnh", label_visibility="collapsed")
        if cam:
            img_input = Image.open(cam)
    else:
        up = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if up:
            img_input = Image.open(up)

    if img_input:
        st.image(img_input, caption="Ảnh đầu vào", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📖 Hướng dẫn Telex")
    st.markdown("""
<div class="guide-box">
Ghép chữ ASL thành từ kiểu Telex:<br>
• <b>Space</b> = kết thúc từ, giải mã Telex<br>
• <b>Del</b> &nbsp;= xóa chữ cuối<br><br>
Ví dụ <b>"xin chào"</b>:<br>
X→I→N→[Space] → C→H→A→O→F→[Space]<br>
⟹ "xin" + "chaof" → <b>xin chào</b>
</div>
""", unsafe_allow_html=True)

# ──── CỘT PHẢI: Kết quả ────────────────────────────────────
with col_result:
    st.markdown("#### 🎯 Kết quả nhận diện")

    if img_input is not None:
        features = extract_landmarks(img_input, hands_detector)

        if features is None:
            st.warning("⚠️ Không phát hiện bàn tay trong ảnh. Hãy thử lại với góc sáng hơn hoặc giơ tay rõ hơn.")
        else:
            proba      = model.predict_proba([features])[0]
            top3_idx   = np.argsort(proba)[::-1][:3]
            top3       = [(CLASS_NAMES[i], float(proba[i])) for i in top3_idx]
            pred_class = top3[0][0]
            confidence = top3[0][1]

            st.session_state.last_pred = pred_class
            st.session_state.last_conf = confidence
            st.session_state.last_top3 = top3

    # Hiển thị kết quả (hoặc placeholder)
    pred  = st.session_state.last_pred or "—"
    conf  = st.session_state.last_conf
    top3  = st.session_state.last_top3

    st.markdown(f"""
<div class="result-box">
    <div class="result-label">Ký hiệu nhận diện</div>
    <div class="result-value">{pred}</div>
    <div class="result-label" style="margin-top:0.6rem">Độ tự tin: {conf*100:.1f}%</div>
    <div class="conf-bar-wrap"><div class="conf-bar" style="width:{conf*100:.0f}%"></div></div>
    {"<div class='top3-row'>" + "".join(f"<div class='top3-item'><div class='top3-letter'>{c}</div><div class='top3-conf'>{p*100:.1f}%</div></div>" for c,p in top3) + "</div>" if top3 else ""}
</div>
""", unsafe_allow_html=True)

    # Nút thao tác
    if st.session_state.last_pred:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(f"➕ Thêm '{pred}'"):
                p = st.session_state.last_pred
                if p == "Space":
                    if st.session_state.buffer:
                        word = telex_decode("".join(st.session_state.buffer).lower())
                        st.session_state.output_words.append(word)
                        st.session_state.buffer = []
                elif p == "Del":
                    if st.session_state.buffer:
                        st.session_state.buffer.pop()
                else:
                    st.session_state.buffer.append(p)
                st.rerun()
        with c2:
            if st.button("⌫ Xóa cuối"):
                if st.session_state.buffer:
                    st.session_state.buffer.pop()
                st.rerun()
        with c3:
            if st.button("␣ Kết từ"):
                if st.session_state.buffer:
                    word = telex_decode("".join(st.session_state.buffer).lower())
                    st.session_state.output_words.append(word)
                    st.session_state.buffer = []
                st.rerun()
    else:
        st.markdown("""
<div class="result-box" style="padding:2.5rem">
    <div style="font-size:2.5rem">🤟</div>
    <div class="result-label" style="margin-top:0.8rem">Chụp hoặc upload ảnh để bắt đầu</div>
</div>""", unsafe_allow_html=True)


# ============================================================
# KHU VỰC GHÉP CHỮ TELEX
# ============================================================
st.markdown("---")
st.markdown("#### ⌨️ Ghép chữ Telex → Tiếng Việt")

buf_str = "".join(st.session_state.buffer)
out_str = " ".join(st.session_state.output_words)

c_buf, c_out = st.columns(2)
with c_buf:
    st.markdown(f"""
<div class="result-box">
    <div class="result-label">Buffer (đang gõ)</div>
    <div class="result-value-viet">{buf_str if buf_str else "_ _ _"}</div>
</div>""", unsafe_allow_html=True)

with c_out:
    st.markdown(f"""
<div class="result-box">
    <div class="result-label">Kết quả tiếng Việt</div>
    <div class="result-value-viet">{out_str if out_str else "_ _ _"}</div>
</div>""", unsafe_allow_html=True)

col_r1, col_r2, _ = st.columns([1, 1, 3])
with col_r1:
    if st.button("🔄 Reset tất cả"):
        st.session_state.buffer       = []
        st.session_state.output_words = []
        st.session_state.last_pred    = None
        st.session_state.last_conf    = 0.0
        st.session_state.last_top3    = []
        st.rerun()
with col_r2:
    if out_str and st.button("📋 Copy kết quả"):
        st.code(out_str)
