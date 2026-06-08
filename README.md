# 🤟 ASL → Tiếng Việt — Nhận diện ngôn ngữ ký hiệu

Ứng dụng nhận diện 27 ký hiệu ASL (A–Z + Space) bằng **MediaPipe + XGBoost**, giao diện **Streamlit**.

## 📊 Kết quả mô hình
| Chỉ số | Giá trị |
|--------|---------|
| Accuracy | **99.3%** |
| Macro F1 | **98.96%** |
| Classes | 27 (A–Z + Space) |

## 🗂️ Cấu trúc repo
```
├── app.py                  # Streamlit app chính
├── model_baseline.pkl      # Model XGBoost đã train (joblib)
├── requirements.txt        # Python dependencies
├── packages.txt            # System packages (OpenCV)
└── README.md
```

## 🚀 Chạy local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy Streamlit Cloud
1. Push repo lên GitHub
2. Vào [share.streamlit.io](https://share.streamlit.io)
3. Chọn repo → `app.py` → Deploy

## 🔧 Pipeline kỹ thuật
```
Ảnh → MediaPipe Hands (21 landmarks × xyz = 63 features)
     → Chuẩn hóa (wrist origin + max-abs scale)
     → XGBoost predict
     → Hiển thị nhãn + độ tự tin
```

## ✍️ Gõ Telex
Ghép chữ cái ASL thành từ kiểu Telex:
- `Space` = kết thúc từ, tự động giải mã Telex
- Ví dụ: `X→I→N→[Space]→C→H→A→O→F→[Space]` → **xin chào**
