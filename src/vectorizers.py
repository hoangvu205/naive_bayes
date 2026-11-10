"""Tạo vectorizer sklearn cho hai cách biểu diễn feature.

Module cung cấp:
- BoW bằng CountVectorizer.
- TF-IDF bằng TfidfVectorizer.
- Hàm chuyển sparse matrix sang list[dict] để manual NB sử dụng.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.manual_nb import tokenize


def make_bow_vectorizer():
    """Tạo CountVectorizer dùng underthesea và unigram + bigram.

    Cấu hình:
    - tokenizer=tokenize: dùng đúng pipeline tách từ của project.
    - token_pattern=None: tắt regex mặc định để không loại bỏ dấu câu.
    - lowercase=False: tokenize đã lowercase, tránh xử lý hai lần.
    - ngram_range=(1, 2): tạo unigram và bigram liền kề.

    Returns:
        CountVectorizer chưa fit.
    """
    return CountVectorizer(  # đếm số lần xuất hiện của feature
        tokenizer=tokenize,  # dùng tokenizer chung của project
        token_pattern=None,  # tắt regex mặc định để giữ dấu câu
        lowercase=False,  # tokenize đã lowercase rồi
        ngram_range=(1, 2),  # tạo unigram và bigram
    )


def make_tfidf_vectorizer():
    """Tạo TfidfVectorizer dùng underthesea và unigram + bigram.

    Cấu hình giống BoW để so sánh công bằng:
    - cùng tokenizer;
    - cùng ngram_range=(1, 2);
    - giữ nguyên dấu câu.

    TF-IDF mặc định dùng:
    - use_idf=True;
    - smooth_idf=True;
    - norm="l2".

    Returns:
        TfidfVectorizer chưa fit.
    """
    return TfidfVectorizer(  # tính trọng số TF-IDF cho feature
        tokenizer=tokenize,  # dùng tokenizer chung của project
        token_pattern=None,  # tắt regex mặc định để giữ dấu câu
        lowercase=False,  # tokenize đã lowercase rồi
        ngram_range=(1, 2),  # tạo unigram và bigram
    )


def features_from_matrix(matrix, vectorizer):
    """Chuyển sparse document-term matrix thành list các mapping feature.

    Mỗi dòng của matrix tương ứng một document. Với mỗi dòng:
    - row.indices chứa index các feature khác 0;
    - row.data chứa giá trị tương ứng (count hoặc TF-IDF weight);
    - vectorizer.get_feature_names_out() ánh xạ index sang tên feature.

    Kết quả là list[dict] để manual NB có thể dùng lại công thức hiện tại
    mà không cần đọc trực tiếp scipy sparse matrix.

    Args:
        matrix: Ma trận sparse do vectorizer.fit_transform/transform trả về.
        vectorizer: Vectorizer đã fit, dùng để lấy tên feature.

    Returns:
        List các dict {feature_name: value}.
    """
    feature_names = vectorizer.get_feature_names_out()  # index -> tên feature
    feature_rows = []  # chứa kết quả cho từng document

    for row_index in range(matrix.shape[0]):  # duyệt từng document
        row = matrix.getrow(row_index)  # lấy dòng sparse tương ứng
        feature_rows.append(  # chuyển các entry khác 0 thành dict
            {
                str(feature_names[column]): value  # tên feature -> count/weight
                for column, value in zip(row.indices, row.data)
            }
        )

    return feature_rows  # danh sách feature của toàn bộ document


def features_from_text(vectorizer, text):
    """Vector hoá một câu và trả về mapping feature của câu đó.

    Hàm dùng vectorizer đã fit để transform text, sau đó lấy dòng đầu tiên
    và chuyển thành dict giống features_from_matrix.

    Args:
        vectorizer: Vectorizer đã fit.
        text: Nội dung review thô.

    Returns:
        Dict {feature_name: value} của câu đầu vào.
    """
    matrix = vectorizer.transform([text])  # vector hoá một câu duy nhất
    return features_from_matrix(matrix, vectorizer)[0]  # lấy dòng đầu tiên
