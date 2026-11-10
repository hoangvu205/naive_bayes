"""Multinomial Naive Bayes viết tay cho phân loại cảm xúc binary.

Module này cung cấp:
- Chuẩn hoá và tokenize tiếng Việt bằng underthesea.
- Train/predict NB trên feature dạng dict hoặc từ ma trận đã vector hoá.
- Lưu/load model bằng joblib.
"""

import math  # dùng log/exp cho NB và softmax
import unicodedata  # chuẩn hoá Unicode NFC
from collections import Counter, defaultdict  # đếm nhãn và feature
from datetime import datetime, timezone  # lưu thời điểm train
from pathlib import Path  # xử lý đường dẫn model

import joblib  # serialize/deserialize model
from underthesea import word_tokenize  # tokenize tiếng Việt

LABELS = ["Negative", "Positive"]  # hai nhãn của bài toán binary
ALPHA = 0.3  # hệ số smoothing mặc định


def normalize_text(text):
    """Chuẩn hoá văn bản trước khi tokenize.

    Các bước:
    1. Ép giá trị về chuỗi.
    2. Chuẩn hoá Unicode NFC để thống nhất cách biểu diễn dấu tiếng Việt.
    3. Chuyển về lowercase và bỏ khoảng trắng đầu/cuối.

    Hàm không xoá dấu câu; underthesea sẽ tách dấu câu thành token riêng.

    Args:
        text: Nội dung review thô.

    Returns:
        Chuỗi đã chuẩn hoá.
    """
    text = str(text)  # đảm bảo đầu vào luôn là chuỗi
    text = unicodedata.normalize("NFC", text)  # thống nhất dấu tiếng Việt
    return text.lower().strip()  # lowercase và bỏ khoảng trắng hai đầu


def tokenize(text):
    """Tách review thành danh sách token bằng underthesea.

    Quy trình:
    1. Chuẩn hoá text bằng normalize_text.
    2. Dùng word_tokenize(format="text") để nhận chuỗi đã tách từ.
    3. split() thành list và bỏ các token rỗng.

    Dấu câu được giữ nguyên theo cấu hình hiện tại.

    Args:
        text: Nội dung review thô.

    Returns:
        Danh sách token; trả về [] nếu text rỗng.
    """
    text = normalize_text(text)  # chuẩn hoá trước khi tách từ
    if not text:  # không xử lý chuỗi rỗng
        return []
    segmented = word_tokenize(text, format="text")  # underthesea trả chuỗi đã tách
    return [token for token in segmented.split() if token]  # bỏ token rỗng


def train_naive_bayes_from_features(feature_rows, labels, alpha=ALPHA):
    """Train Multinomial Naive Bayes từ các feature row đã có sẵn.

    Args:
        feature_rows: Iterable các mapping {feature: value}.
            value có thể là count (BoW) hoặc trọng số (TF-IDF).
        labels: Nhãn tương ứng với từng feature row.
        alpha: Hệ số smoothing; alpha=1 là Laplace, alpha=0.3 là Lidstone.

    Returns:
        Dict model chứa class prior, feature counts, tổng feature value,
        vocab size và metadata để predict tính xác suất sau này.

    Cách hoạt động:
    1. Đếm số document của mỗi lớp để tính class prior.
    2. Cộng dồn value của từng feature theo từng lớp.
    3. Cộng tổng value của tất cả feature theo lớp; giá trị này là mẫu số
       cho công thức xác suất có smoothing.
    4. Lưu vocab size để áp dụng công thức:
       P(t|c) = (count(t,c) + alpha) / (total_features(c) + alpha * V).
    """
    feature_rows = list(feature_rows)  # vật chất hoá iterable để dùng nhiều lần
    labels = list(labels)  # vật chất hoá nhãn để kiểm tra độ dài
    if not feature_rows:  # không thể train với dữ liệu rỗng
        raise ValueError("Cannot train on an empty dataset")
    if len(feature_rows) != len(labels):  # số dòng feature phải khớp số nhãn
        raise ValueError("feature_rows and labels must have the same length")

    classes = sorted(set(labels))  # danh sách lớp theo thứ tự cố định
    class_counts = Counter(labels)  # N_c: số document của từng lớp
    feature_counts = {label: defaultdict(float) for label in classes}  # count/weight
    total_features = {label: 0.0 for label in classes}  # tổng value mỗi lớp
    vocabulary = set()  # tập feature để tính V

    for features, label in zip(feature_rows, labels):  # duyệt từng document
        for feature, value in features.items():  # duyệt feature khác 0
            feature_counts[label][feature] += value  # cộng dồn vào lớp
            total_features[label] += value  # cộng vào mẫu số smoothing
            vocabulary.add(feature)  # ghi nhận feature vào vocab

    return {
        "format": "manual-multinomial-nb",  # định danh loại model
        "version": 1,  # phiên bản schema model
        "trained_at": datetime.now(timezone.utc).isoformat(),  # thời điểm train
        "alpha": float(alpha),  # smoothing dùng khi predict
        "ngram_range": (1, 2),  # model dùng unigram + bigram
        "classes": classes,  # danh sách nhãn
        "class_counts": dict(class_counts),  # số document mỗi lớp
        "num_documents": len(feature_rows),  # tổng số document
        "feature_counts": {  # count/weight từng feature theo lớp
            label: dict(counts) for label, counts in feature_counts.items()
        },
        "total_features": total_features,  # tổng value từng lớp
        "vocab_size": len(vocabulary),  # V trong công thức smoothing
    }


def predict_log_scores_from_features(model, features):
    """Tính log score của từng lớp cho một document đã vector hoá.

    Công thức:
        log_score(c) = log P(c) + sum(value(t) * log P(t|c))

        P(t|c) = (count(t,c) + alpha) / (total_features(c) + alpha * V)

    Dùng log để tránh underflow khi nhân nhiều xác suất nhỏ và để cộng dồn
    đóng góp của từng feature.

    Args:
        model: Dict model do train_naive_bayes_from_features trả về.
        features: Mapping {feature: value} của một document.

    Returns:
        Dict {label: log_score}.
    """
    scores = {}  # log score của từng lớp

    for label in model["classes"]:  # tính score riêng cho từng lớp
        total_features = model["total_features"][label]  # tổng value của lớp
        denominator = (  # mẫu số xác suất có smoothing
            total_features + model["alpha"] * model["vocab_size"]
        )
        log_prior = math.log(  # log class prior: log(N_c / N)
            model["class_counts"][label] / model["num_documents"]
        )
        score = log_prior  # score bắt đầu bằng log prior
        class_feature_counts = model["feature_counts"][label]  # count theo lớp

        for feature, value in features.items():  # duyệt feature của document
            feature_count = class_feature_counts.get(feature, 0)  # count trong lớp
            probability = (  # P(t|c) có smoothing
                feature_count + model["alpha"]
            ) / denominator
            score += value * math.log(probability)  # cộng log likelihood

            # Công thức gốc không smoothing (để tham khảo, không dùng):
            # probability = feature_count / total_features if feature_count else 0.0
            # score += value * (
            #     math.log(probability) if probability > 0 else float("-inf")
            # )

        scores[label] = score  # lưu score của lớp hiện tại

    return scores  # trả về log score từng lớp


def predict_from_features(model, features):
    """Chuyển log score thành nhãn, xác suất và score trong [-1, 1].

    Các bước:
    1. Tính log score cho từng lớp.
    2. Chọn lớp có log score lớn nhất làm nhãn dự đoán.
    3. Chuyển log score sang xác suất bằng stable softmax: trừ max trước
       khi gọi exp để tránh tràn số.
    4. Tính score = P(Positive) - P(Negative) và clip về [-1, 1].

    Args:
        model: Dict model đã train.
        features: Mapping {feature: value} của một document.

    Returns:
        Tuple (label, probabilities, score).
    """
    scores = predict_log_scores_from_features(model, features)  # log score
    best_label = max(scores, key=scores.get)  # nhãn có score cao nhất
    max_score = max(scores.values())  # dùng cho stable softmax
    weights = {  # exp(score - max) để tránh tràn số
        label: math.exp(score - max_score) for label, score in scores.items()
    }
    total = sum(weights.values())  # tổng trọng số sau exp
    probabilities = {  # chuẩn hoá thành xác suất
        label: weights[label] / total for label in model["classes"]
    }
    score = probabilities.get("Positive", 0.0) - probabilities.get(  # score
        "Negative", 0.0
    )
    score = max(-1.0, min(1.0, score))  # đảm bảo score nằm trong [-1, 1]
    return best_label, probabilities, score  # trả về kết quả cuối


def save_model(model, path):
    """Lưu model hoặc model bundle bằng joblib.

    - Tự tạo thư mục cha nếu chưa tồn tại.
    - Dùng compress=3 để giảm dung lượng file.
    - joblib dựa trên pickle, nên chỉ nên load lại file do mình tạo.

    Args:
        model: Dict model hoặc bundle cần lưu.
        path: Đường dẫn file đầu ra.
    """
    path = Path(path)  # đảm bảo path là đối tượng Path
    path.parent.mkdir(parents=True, exist_ok=True)  # tạo thư mục nếu thiếu
    joblib.dump(model, path, compress=3)  # ghi model nén xuống đĩa


def load_model(path):
    """Load model hoặc model bundle đã lưu bằng joblib.

    Cảnh báo: joblib/pickle có thể thực thi code khi load. Không load file
    từ nguồn không tin cậy.

    Args:
        path: Đường dẫn file model.

    Returns:
        Model hoặc bundle đã lưu.
    """
    return joblib.load(path)  # đọc model từ đĩa vào bộ nhớ
