"""Train và so sánh hai phiên bản feature cho manual Naive Bayes.

Script thực hiện:
1. Đọc và làm sạch dataset_labeled.csv.
2. Chia train/test 80/20, stratify theo nhãn.
3. Train/evaluate BoW và TF-IDF trên cùng một split.
4. Lưu confusion matrix cho từng phiên bản.
5. Fit lại trên toàn bộ dữ liệu và lưu 2 model bundle.
"""

import sys  # thêm project root vào sys.path
from pathlib import Path  # xử lý đường dẫn

import matplotlib  # thư viện vẽ biểu đồ

matplotlib.use("Agg")  # dùng backend không cần giao diện
import matplotlib.pyplot as plt  # vẽ confusion matrix
import pandas as pd  # đọc CSV và tạo bảng
from sklearn.metrics import (  # các metric đánh giá
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split  # chia train/test

if hasattr(sys.stdout, "reconfigure"):  # tránh lỗi encoding trên Windows
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]  # thư mục gốc project
if str(ROOT) not in sys.path:  # đảm bảo import được package src
    sys.path.insert(0, str(ROOT))

from src.manual_nb import (  #NB và hàm lưu model
    ALPHA,
    LABELS,
    predict_from_features,
    save_model,
    train_naive_bayes_from_features,
)
from src.vectorizers import (  # hai loại vectorizer
    features_from_matrix,
    make_bow_vectorizer,
    make_tfidf_vectorizer,
)

DATA_PATH = ROOT / "dataset_labeled.csv"  # dữ liệu train
MODEL_DIR = ROOT / "models"  # thư mục lưu model
REPORT_DIR = ROOT / "reports"  # thư mục lưu ảnh/report
RANDOM_STATE = 42  # seed để tái lập kết quả
VECTORIZERS = [  # danh sách phiên bản cần train và so sánh
    ("bow", "BoW", make_bow_vectorizer),
    ("tfidf", "TF-IDF", make_tfidf_vectorizer),
]


def save_confusion_matrix(matrix, title, path):
    """Vẽ và lưu confusion matrix thành file ảnh PNG.

    Args:
        matrix: Ma trận confusion dạng numpy array.
        title: Tiêu đề hiển thị trên ảnh.
        path: Đường dẫn file PNG đầu ra.

    Hàm dùng backend Agg để chạy được trong môi trường không có giao diện,
    sau đó đóng figure để giải phóng bộ nhớ.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)  # tạo thư mục reports nếu thiếu
    display = ConfusionMatrixDisplay(  # đối tượng hiển thị confusion matrix
        confusion_matrix=matrix, display_labels=LABELS
    )
    display.plot(cmap="Blues", values_format="d", colorbar=False)  # vẽ ma trận
    plt.title(title)  # gắn tiêu đề cho ảnh
    plt.tight_layout()  # căn chỉnh để không cắt nhãn
    plt.savefig(path, dpi=150)  # lưu ảnh PNG độ phân giải 150 dpi
    plt.close()  # đóng figure để giải phóng bộ nhớ


def evaluate_model(name, model, test_features, y_test, confusion_path):
    """Đánh giá một model trên tập test và lưu confusion matrix.

    Args:
        name: Tên hiển thị của phiên bản, ví dụ "BoW" hoặc "TF-IDF".
        model: Dict manual NB đã train.
        test_features: List các mapping feature của tập test.
        y_test: Nhãn thật của tập test.
        confusion_path: Đường dẫn ảnh confusion matrix.

    Returns:
        Dict tóm tắt accuracy, macro-F1, weighted-F1 và F1 từng lớp
        để dùng cho bảng so sánh cuối script.
    """
    predictions = [  # dự đoán nhãn cho từng document test
        predict_from_features(model, features)[0]
        for features in test_features
    ]
    accuracy = accuracy_score(y_test, predictions)  # tỷ lệ dự đoán đúng
    report_text = classification_report(  # report dạng text để in console
        y_test,
        predictions,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )
    report = classification_report(  # report dạng dict để lấy metric
        y_test,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)  # ma trận lỗi

    print(f"\n=== {name} ===")  # in tên phiên bản đang đánh giá
    print(f"Vocabulary size: {model['vocab_size']}")  # số feature của model
    print(f"Accuracy: {accuracy:.4f}")  # in accuracy
    print("\nClassification report:")  # tiêu đề report
    print(report_text)  # in precision/recall/F1 từng lớp
    print("Confusion matrix (rows=true, cols=predicted):")  # tiêu đề ma trận
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS))  # in ma trận
    save_confusion_matrix(matrix, f"Manual Naive Bayes - {name}", confusion_path)
    print(f"Saved confusion matrix: {confusion_path}")  # báo đường dẫn ảnh

    return {  # trả về số liệu tóm tắt để so sánh
        "name": name,
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "negative_f1": report[LABELS[0]]["f1-score"],
        "positive_f1": report[LABELS[1]]["f1-score"],
    }


def main():
    """Chạy toàn bộ pipeline train, evaluate và lưu model.

    Luồng xử lý:
    1. Đọc CSV và chỉ giữ review_text/sentiment hợp lệ.
    2. Chia train/test với cùng random_state và stratify.
    3. Với mỗi vectorizer, train trên train set, đánh giá trên test set.
    4. Fit lại vectorizer + NB trên toàn bộ dữ liệu để lưu model cuối.
    5. In bảng so sánh metrics giữa BoW và TF-IDF.
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")  # đọc CSV
    df = df[["text", "label"]].dropna()
    df = df.rename(columns={"text": "review_text"})
    df["sentiment"] = df["label"].map({1: "Positive", -1: "Negative"})
    df = df[["review_text", "sentiment"]].dropna()  # bỏ dòng thiếu text/nhãn
    df["review_text"] = df["review_text"].astype(str).str.strip()  # chuẩn hoá text
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.title()  # chuẩn hoá nhãn
    df = df[df["sentiment"].isin(LABELS) & df["review_text"].str.len().gt(0)]  # lọc hợp lệ
    df = df.drop_duplicates("review_text").reset_index(drop=True)  # bỏ review trùng

    X_train, X_test, y_train, y_test = train_test_split(  # chia train/test
        df["review_text"],
        df["sentiment"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"],
    )

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")  # kích thước split
    results = []  # lưu metric của từng phiên bản

    # Dùng cùng train/test split cho cả BoW và TF-IDF để so sánh công bằng.
    for key, name, make_vectorizer in VECTORIZERS:  # duyệt từng phiên bản
        vectorizer = make_vectorizer()  # tạo vectorizer mới
        # Fit vocabulary trên train set, chỉ transform test set để tránh leak.
        train_matrix = vectorizer.fit_transform(X_train.tolist())  # fit + encode train
        test_matrix = vectorizer.transform(X_test.tolist())  # encode test
        train_features = features_from_matrix(train_matrix, vectorizer)  # matrix -> dict
        test_features = features_from_matrix(test_matrix, vectorizer)  # matrix -> dict

        model = train_naive_bayes_from_features(  # train NB trên train set
            train_features, y_train.tolist(), alpha=ALPHA
        )
        results.append(  # đánh giá và lưu kết quả
            evaluate_model(
                name,
                model,
                test_features,
                y_test.tolist(),
                REPORT_DIR / f"confusion_matrix_{key}.png",
            )
        )

        # Sau khi đánh giá, fit lại vectorizer + NB trên toàn bộ dữ liệu
        # để model lưu ra dùng cho predict.
        final_vectorizer = make_vectorizer()  # vectorizer mới cho full data
        full_matrix = final_vectorizer.fit_transform(  # fit + encode toàn bộ
            df["review_text"].tolist()
        )
        full_features = features_from_matrix(full_matrix, final_vectorizer)  # matrix -> dict
        final_model = train_naive_bayes_from_features(  # train NB full data
            full_features, df["sentiment"].tolist(), alpha=ALPHA
        )
        bundle = {  # đóng gói vectorizer + model để lưu
            "format": "manual-multinomial-nb-bundle",
            "version": 1,
            "vectorizer_name": key,
            "vectorizer": final_vectorizer,
            "model": final_model,
        }
        model_path = MODEL_DIR / f"manual_nb_{key}.joblib"  # đường dẫn model
        save_model(bundle, model_path)  # lưu bundle bằng joblib
        print(f"Saved model: {model_path}")  # báo đường dẫn model

    # In bảng tổng hợp để so sánh nhanh hai phiên bản feature.
    print("\nComparison:")  # tiêu đề bảng so sánh
    print(  # in header bảng
        f"{'Input':<8} {'Accuracy':>10} {'Macro F1':>10} "
        f"{'Weighted F1':>12} {'Neg F1':>8} {'Pos F1':>8}"
    )
    for result in results:  # in từng dòng kết quả
        print(
            f"{result['name']:<8} "
            f"{result['accuracy']:>10.4f} "
            f"{result['macro_f1']:>10.4f} "
            f"{result['weighted_f1']:>12.4f} "
            f"{result['negative_f1']:>8.4f} "
            f"{result['positive_f1']:>8.4f}"
        )


if __name__ == "__main__":  # chỉ chạy khi gọi trực tiếp script
    main()
