"""Dự đoán cảm xúc bằng model TF-IDF đã train.

Script load bundle TfidfVectorizer + manual NB, vector hoá câu nhập vào,
sau đó in nhãn, xác suất từng lớp và score trong [-1, 1].
"""

import sys  # đọc argument từ CLI
from pathlib import Path  # xử lý đường dẫn model

if hasattr(sys.stdout, "reconfigure"):  # tránh lỗi encoding trên Windows
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]  # thư mục gốc project
if str(ROOT) not in sys.path:  # đảm bảo import được package src
    sys.path.insert(0, str(ROOT))

from src.manual_nb import LABELS, load_model, predict_from_features  # NB helpers
from src.vectorizers import features_from_text  # vector hoá một câu

MODEL_PATH = ROOT / "models" / "manual_nb_tfidf.joblib"  # model TF-IDF cần load


def main():
    """Load model TF-IDF, dự đoán câu nhập từ CLI và in kết quả.

    Luồng xử lý:
    1. Ghép các argument thành một câu review.
    2. Load bundle joblib chứa vectorizer và manual NB.
    3. Vector hoá câu bằng TfidfVectorizer đã fit.
    4. Dự đoán nhãn, xác suất và score.
    """
    if len(sys.argv) < 2:  # bắt buộc phải có nội dung review
        raise SystemExit('Usage: python scripts/predict_tfidf.py "nội dung review"')
    if not MODEL_PATH.exists():  # kiểm tra model đã được train chưa
        raise SystemExit(f"Model not found: {MODEL_PATH}. Run scripts/train.py first.")

    text = " ".join(sys.argv[1:]).strip()  # ghép argument thành một câu
    bundle = load_model(MODEL_PATH)  # load vectorizer + model
    features = features_from_text(bundle["vectorizer"], text)  # câu -> feature
    label, probabilities, score = predict_from_features(  # dự đoán
        bundle["model"], features
    )

    print(f"Text: {text}")  # in câu đã nhập
    print(f"Label: {label}")  # in nhãn dự đoán
    for class_name in LABELS:  # in xác suất từng lớp
        print(f"  {class_name}: {probabilities[class_name]:.4f}")
    print(f"Score: {score:+.4f}  (range -1..1)")  # in score cuối


if __name__ == "__main__":  # chỉ chạy khi gọi trực tiếp script
    main()
