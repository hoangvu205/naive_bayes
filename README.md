# Naive Bayes code tay + BoW/TF-IDF cho đánh giá sản phẩm

Mô hình phân loại cảm xúc binary (`Negative`, `Positive`) với Multinomial Naive Bayes viết tay.

- Dữ liệu train: `dataset_labeled.csv`.
- File dữ liệu có các cột: `id`, `category`, `text`, `label`, `confidence`.
- Pipeline map `text` → `review_text`, `label 1` → `Positive`, `label -1` → `Negative`.
- Chỉ dùng `text` và `label` để train; `id`, `category`, `confidence` hiện chưa được dùng.
- Dataset hiện tại không có cột `star`.
- Tokenize bằng `underthesea`, giữ nguyên dấu câu.
- Feature: unigram + bigram liền kề.
- Không bỏ stopword.
- Có smoothing `alpha=0.3`.
- So sánh 2 cách vector hoá:
  - BoW: `sklearn.feature_extraction.text.CountVectorizer`
  - TF-IDF: `sklearn.feature_extraction.text.TfidfVectorizer`
- Chia train/test 80/20 ngẫu nhiên, stratify theo nhãn.
- Đánh giá bằng accuracy, precision, recall, F1-score và confusion matrix.

## Dữ liệu

`dataset_labeled.csv` hiện có:

| Hạng mục | Giá trị |
|---|---:|
| Tổng số dòng | 6.938 |
| Positive (`label=1`) | 4.088 |
| Negative (`label=-1`) | 2.850 |
| Dòng thiếu/rỗng `text` | 0 |
| Review trùng theo `text` | 0 |

Vì dữ liệu đã sạch và không có duplicate, bước bỏ trùng trong `scripts/train.py` không làm giảm thêm số dòng.

## Cài đặt

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Huấn luyện

```bash
.venv/Scripts/python.exe -X utf8 scripts/train.py
```

Script sẽ:

1. Đọc `dataset_labeled.csv`, map `text`/`label` sang `review_text`/`sentiment`.
2. Bỏ dòng thiếu dữ liệu, chỉ giữ `Negative`/`Positive`, bỏ duplicate `review_text`.
3. Chia 80/20.
4. Train và đánh giá lần lượt BoW và TF-IDF.
5. Lưu ảnh confusion matrix cho từng phiên bản.
6. Fit lại vectorizer + NB trên toàn bộ dữ liệu và lưu 2 model.

Cấu hình cố định:

```text
alpha=0.3
ngram_range=(1, 2)
tokenizer=underthesea
punctuation=giữ nguyên
```

Output:

```text
models/manual_nb_bow.joblib
models/manual_nb_tfidf.joblib
reports/confusion_matrix_bow.png
reports/confusion_matrix_tfidf.png
```

Mỗi model là một bundle `joblib` (nén `compress=3`) gồm vectorizer, manual NB model và metadata.

## Kết quả đánh giá

Kết quả trên tập test 80/20, `random_state=42`, stratify theo nhãn:

```text
Train rows: 5550 | Test rows: 1388
Vocabulary train: 63202
```

| Input | Accuracy | Macro F1 | Weighted F1 | Neg Precision | Neg Recall | Neg F1 | Pos Precision | Pos Recall | Pos F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BoW | 0.9409 | 0.9389 | 0.9409 | 0.9296 | 0.9263 | 0.9279 | 0.9488 | 0.9511 | 0.9499 |
| TF-IDF | 0.9438 | 0.9420 | 0.9438 | 0.9301 | 0.9333 | 0.9317 | 0.9534 | 0.9511 | 0.9523 |

Confusion matrix (hàng = nhãn thật, cột = nhãn dự đoán):

```text
BoW
          Negative  Positive
Negative       528        42
Positive        40       778

TF-IDF
          Negative  Positive
Negative       532        38
Positive        40       778
```

![Confusion matrix BoW](reports/confusion_matrix_bow.png)

![Confusion matrix TF-IDF](reports/confusion_matrix_tfidf.png)

### Nhận xét chi tiết BoW vs TF-IDF

#### 1. Chênh lệch tổng quan

| Chỉ số | BoW | TF-IDF | Chênh lệch |
|---|---:|---:|---:|
| Accuracy | 0.9409 | 0.9438 | +0.0029 |
| Macro F1 | 0.9389 | 0.9420 | +0.0031 |
| Weighted F1 | 0.9409 | 0.9438 | +0.0029 |

- TF-IDF nhỉnh hơn BoW nhưng mức chênh lệch rất nhỏ, khoảng **0,29 điểm phần trăm accuracy**.
- Với 1.388 mẫu test, mức tăng này tương đương khoảng **4 mẫu dự đoán đúng thêm**.
- Chênh lệch nằm trong một lần chia train/test, chưa có cross-validation hoặc kiểm định ý nghĩa thống kê, nên chưa thể kết luận TF-IDF luôn tốt hơn.

#### 2. Phân tích theo lớp

**Lớp Negative**

| Chỉ số | BoW | TF-IDF | Nhận xét |
|---|---:|---:|---|
| Precision | 0.9296 | 0.9301 | TF-IDF tốt hơn rất nhẹ |
| Recall | 0.9263 | 0.9333 | TF-IDF tốt hơn |
| F1 | 0.9279 | 0.9317 | TF-IDF tốt hơn nhẹ |

- BoW dự đoán đúng **528** review Negative; TF-IDF dự đoán đúng **532**.
- TF-IDF giảm số review Negative bị đoán nhầm thành Positive từ **42 xuống 38**.
- Số review Positive bị đoán nhầm thành Negative giữ nguyên là **40** ở cả hai model.
- Nhờ 4 review Negative được dự đoán đúng thêm, recall Negative tăng từ `0.9263` lên `0.9333`.

**Lớp Positive**

| Chỉ số | BoW | TF-IDF | Nhận xét |
|---|---:|---:|---|
| Precision | 0.9488 | 0.9534 | TF-IDF tốt hơn nhẹ |
| Recall | 0.9511 | 0.9511 | Không đổi |
| F1 | 0.9499 | 0.9523 | TF-IDF tốt hơn nhẹ |

- Cả hai model đều dự đoán đúng **778** review Positive.
- Cả hai đều dự đoán sai **40** review Positive thành Negative.
- TF-IDF giảm số Negative bị đoán nhầm thành Positive từ **42 xuống 38**, nhờ đó precision của Positive tăng.
- Recall Positive không cải thiện vì số Positive bị đoán thành Negative vẫn giữ nguyên là **40**.

Tóm lại, TF-IDF cải thiện chủ yếu ở việc **giảm Negative bị dự đoán thành Positive** từ **42 xuống 38**, qua đó tăng nhẹ recall Negative và precision Positive. Điểm yếu chung của cả hai là **40 review Positive vẫn bị dự đoán thành Negative**, tương ứng recall Positive chỉ khoảng `0.9511`.

#### 3. Vì sao có khác biệt

- **BoW** dùng số lần xuất hiện thô. Từ nào xuất hiện nhiều sẽ có trọng số lớn, kể cả các từ phổ biến nhưng ít phân biệt.
- **TF-IDF** giảm trọng số các từ xuất hiện ở nhiều văn bản và tăng trọng số các từ hiếm, có tính phân biệt cao hơn.
- `TfidfVectorizer` mặc định chuẩn hoá L2, nên review dài không bị chi phối quá mạnh bởi số lượng từ.
- Với BoW, manual NB gần với Multinomial NB trên count thuần. Với TF-IDF, manual NB trở thành **weighted NB** trên trọng số TF-IDF, không còn đúng giả định xác suất Multinomial NB ban đầu.
- Vì vậy, score của TF-IDF không nên được diễn giải như xác suất Multinomial NB chuẩn; chỉ nên dùng để so sánh tương đối.

#### 4. Ví dụ định tính

```text
"giao hàng nhanh nhưng hàng quá tệ"
BoW    -> Positive, score +0.1932
TF-IDF -> Positive, score +0.3244
```

- Cả hai đều dự đoán Positive, nhưng BoW ít tự tin hơn với score `+0.1932`.
- Câu có cả ý tích cực và tiêu cực vẫn là trường hợp khó; cả hai model đều nghiêng về cụm tích cực `giao hàng nhanh`.
- Không nên dùng một câu đơn lẻ để kết luận model nào tốt hơn.

```text
"pin dùng tốt giao hàng nhanh"
BoW    -> Positive, score +1.0000
TF-IDF -> Positive, score +0.9896
```

- Cả hai đều đúng nhãn, nhưng BoW tự tin gần như tuyệt đối; TF-IDF cho score bớt cực đoan hơn.

Thêm ba ví dụ theo mức score:

| Mức score | Câu review | Nhãn | BoW | TF-IDF |
|---|---|---|---:|---:|
| Gần 0 | "Làm lạnh ổn. Thợ lắp k hút chân không," | Positive | `-0.1489` | `-0.0765` |
| Gần +1 | "Sản phẩm đẹp, giá cả hợp lý, hâm nóng thấy rất nhanh và tiện dụng, máy chạy êm. Nhân viên hướng dẫn rất nhiệt tình, dễ hiểu." | Positive | `+1.0000` | `+1.0000` |
| Gần -1 | "Mới lắp được mấy ngày. Ngày thứ 2 bật lên thì bị chảy nước. Gọi bảo hành k được. Bên ĐMX lắp đặt kém quá vậy." | Negative | `-1.0000` | `-0.9995` |

- Score gần `0` cho thấy model không nghiêng rõ về Positive hay Negative; thường gặp ở câu ngắn, câu pha trộn cảm xúc hoặc thiếu tín hiệu phân biệt.
- Score gần `+1` là Positive rất mạnh; score gần `-1` là Negative rất mạnh.
- Score chỉ nên dùng để so sánh tương đối, không phải xác suất đã hiệu chỉnh; với TF-IDF, đây là score của weighted NB.

#### 5. Kết luận

- Nếu ưu tiên **accuracy trên split hiện tại**: chọn TF-IDF.
- Nếu ưu tiên **đơn giản, dễ giải thích và đúng Multinomial NB hơn**: chọn BoW.
- Chênh lệch hiện tại chưa đủ lớn để nói TF-IDF vượt trội rõ rệt.
- Để kết luận chắc hơn nên:
  - chạy cross-validation,
  - thử nhiều `alpha` và `ngram_range`,
  - phân tích các câu dự đoán sai,
  - bổ sung xử lý phủ định và cụm đối lập,
  - hiệu chỉnh ngưỡng dự đoán nếu cần ưu tiên precision hoặc recall.

## Dự đoán

BoW:

```bash
.venv/Scripts/python.exe -X utf8 scripts/predict_bow.py "Pin dùng rất tốt, giao hàng nhanh"
```

TF-IDF:

```bash
.venv/Scripts/python.exe -X utf8 scripts/predict_tfidf.py "Pin dùng rất tốt, giao hàng nhanh"
```

Kết quả gồm nhãn dự đoán, xác suất `Negative`/`Positive` và `score` trong `[-1, 1]`:

```text
score = P(Positive) - P(Negative)
```

TF-IDF + manual NB là weighted NB trên trọng số TF-IDF, không còn đúng Multinomial NB trên count thuần.

## Ghi chú về dữ liệu

- `dataset_labeled.csv` có cột `confidence`, nhưng pipeline hiện không dùng cột này để lọc hay tính trọng số.
- Chưa có thông tin xác nhận nhãn được kiểm tra bởi con người, nên metrics chỉ phản ánh chất lượng trên dataset hiện tại.
- `reviews.csv`, `reviews.jsonl`, `reviews_labeled.csv` và `nhap.py` là dữ liệu/script cũ, không còn là nguồn train hiện tại.

## Cấu trúc

```text
dataset_labeled.csv  # nguồn train hiện tại
models/
  manual_nb_bow.joblib
  manual_nb_tfidf.joblib
reports/
  confusion_matrix_bow.png
  confusion_matrix_tfidf.png
src/
  manual_nb.py       # Multinomial NB code tay
  vectorizers.py     # CountVectorizer và TfidfVectorizer
scripts/
  train.py           # train, đánh giá, lưu 2 model
  predict_bow.py     # dự đoán bằng BoW
  predict_tfidf.py   # dự đoán bằng TF-IDF
```
