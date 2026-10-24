from collections import Counter
from io import StringIO
from pathlib import Path
import csv
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DEMO_PATH = ROOT / "data" / "demo_transactions.csv"
NORMAL_COLOR = "#2563EB"
FRAUD_COLOR = "#E45454"


@st.cache_resource
def load_artifacts(models_dir=MODELS_DIR):
    paths = [models_dir / name for name in ["best_model.joblib", "threshold.json", "features.json"]]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu file {path.name} trong thư mục models. Hãy hoàn thành giai đoạn 4.")
    model = joblib.load(paths[0])
    with paths[1].open(encoding="utf-8") as file:
        threshold = json.load(file)["threshold"]
    with paths[2].open(encoding="utf-8") as file:
        features = json.load(file)
    if not isinstance(features, list) or len(features) != 30 or not all(isinstance(name, str) for name in features):
        raise ValueError("features.json phải chứa danh sách 30 tên đặc trưng.")
    if len(set(features)) != 30 or "Class" in features:
        raise ValueError("Danh sách đặc trưng bị trùng hoặc chứa cột Class.")
    if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Ngưỡng trong threshold.json phải là số từ 0 đến 1.")
    if not isinstance(model, RandomForestClassifier) or model.feature_names_in_.tolist() != features:
        raise ValueError("Mô hình hoặc thứ tự đặc trưng không khớp các file của giai đoạn 4.")
    if model.classes_.tolist() != [0, 1]:
        raise ValueError("Mô hình phải có hai nhãn Class 0 và Class 1.")
    return model, float(threshold), features


def read_transactions(payload, features):
    try:
        text = payload.decode("utf-8-sig")
        reader = csv.reader(StringIO(text), strict=True)
        header = next(reader, [])
        if not header:
            raise ValueError("File CSV rỗng.")
        duplicate_names = [name for name, count in Counter(name.strip() for name in header).items() if count > 1]
        if duplicate_names:
            raise ValueError("Tên cột bị trùng: " + ", ".join(duplicate_names))
        for row_number, row in enumerate(reader, start=2):
            if row and len(row) != len(header):
                raise ValueError(f"Dòng CSV {row_number} có số cột không khớp dòng tiêu đề.")
        extra_columns = {name: "string" for name in header if name not in features and name != "Class"}
        frame = pd.read_csv(StringIO(text), dtype=extra_columns, float_precision="round_trip")
    except UnicodeDecodeError as error:
        raise ValueError("Không đọc được mã hóa. Hãy lưu CSV dưới dạng UTF-8.") from error
    except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as error:
        raise ValueError("Không đọc được CSV. Hãy kiểm tra dấu phẩy phân cách và dấu ngoặc kép.") from error
    if frame.empty:
        raise ValueError("File chưa có giao dịch để dự đoán.")
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError("Thiếu cột đầu vào: " + ", ".join(missing))
    reserved = [name for name in ["fraud_score", "prediction"] if name in frame.columns]
    if reserved:
        raise ValueError("File đã có cột kết quả " + ", ".join(reserved) + ". Hãy tải CSV giao dịch gốc.")
    non_numeric = [name for name in features if not is_numeric_dtype(frame[name]) or is_bool_dtype(frame[name])]
    if non_numeric:
        raise ValueError("Các cột sau phải chứa dữ liệu số: " + ", ".join(non_numeric))
    if frame[features].isna().any().any():
        raise ValueError("Các đặc trưng có giá trị thiếu. Hãy kiểm tra lại CSV trước khi dự đoán.")
    values = frame[features].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Các đặc trưng có giá trị vô hạn (inf/-inf) hoặc không hợp lệ.")
    if np.abs(values).max() > np.finfo(np.float32).max:
        raise ValueError("Có giá trị quá lớn để mô hình xử lý. Hãy kiểm tra lại CSV.")
    return frame


def predict_transactions(frame, model, features, threshold):
    scores = model.predict_proba(frame.loc[:, features])[:, 1]
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Mô hình trả về điểm dự đoán không hợp lệ.")
    result = frame.copy()
    result["fraud_score"] = scores
    result["prediction"] = (scores >= threshold).astype(int)
    return result


def evaluate_labels(result):
    if "Class" not in result.columns:
        return None, None
    labels = result["Class"]
    if not is_numeric_dtype(labels) or is_bool_dtype(labels) or labels.isna().any() or not labels.isin([0, 1]).all():
        return None, "Cột Class không hợp lệ: cần nhãn số 0 hoặc 1 cho mọi dòng. Phần dự đoán vẫn hoạt động; bỏ qua đánh giá."
    if labels.nunique() < 2:
        return None, "Dữ liệu chỉ có một lớp thực tế. Cần cả Class 0 và Class 1 để hiển thị đầy đủ Precision, Recall và F1."
    predictions = result["prediction"]
    metrics = {
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1-score": f1_score(labels, predictions, zero_division=0),
    }
    return {"metrics": metrics, "matrix": confusion_matrix(labels, predictions, labels=[0, 1])}, None


def render_results(result, filename, threshold, download_data):
    total = len(result)
    flagged = int(result["prediction"].sum())
    st.subheader("Tổng quan kết quả")
    st.caption(f"{filename} · {total:,} giao dịch · Ngưỡng đang áp dụng: {threshold:.2f}")
    for column, label, value in zip(
        st.columns(4),
        ["Tổng giao dịch", "Bình thường", "Nghi gian lận", "Tỷ lệ cảnh báo"],
        [f"{total:,}", f"{total - flagged:,}", f"{flagged:,}", f"{flagged / total * 100:.2f}%"],
    ):
        column.metric(label, value, border=True)

    chart_column, detail_column = st.columns([1, 1.3], gap="large")
    with chart_column:
        with st.container(border=True):
            st.markdown("**Phân bố kết quả dự đoán**")
            figure = go.Figure(go.Pie(
                labels=["Bình thường", "Nghi gian lận"], values=[total - flagged, flagged],
                hole=0.72, marker_colors=[NORMAL_COLOR, FRAUD_COLOR],
                textinfo="percent", sort=False,
                hovertemplate="%{label}: %{value:,} giao dịch (%{percent})<extra></extra>",
            ))
            figure.update_layout(
                height=310, margin=dict(l=15, r=15, t=15, b=15),
                legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
                annotations=[dict(text=f"<b>{total:,}</b><br>giao dịch", x=0.5, y=0.5, showarrow=False, font_size=20)],
                paper_bgcolor="rgba(0,0,0,0)", font_family="sans-serif",
            )
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    with detail_column:
        with st.container(border=True):
            st.markdown("**Đọc kết quả**")
            st.markdown("**Bình thường · Class 0**  \nĐiểm dự đoán thấp hơn ngưỡng phân loại.")
            st.markdown("**Nghi gian lận · Class 1**  \nĐiểm dự đoán đạt hoặc vượt ngưỡng, cần kiểm tra thêm giao dịch.")
            st.caption("Điểm mô hình không phải kết luận chắc chắn về gian lận. Ứng dụng minh họa mô hình đã lưu, không huấn luyện hoặc thay đổi ngưỡng.")

    st.subheader("Chi tiết giao dịch")
    filter_column, sort_column = st.columns([2, 1])
    group = filter_column.selectbox("Nhóm giao dịch", ["Tất cả giao dịch", "Bình thường", "Nghi gian lận"])
    sort_scores = sort_column.checkbox("Điểm dự đoán giảm dần", value=True)
    display_table = pd.DataFrame({
        "Dòng dữ liệu": np.arange(1, total + 1),
        "Amount": result["Amount"].to_numpy(),
        "fraud_score": result["fraud_score"].to_numpy(),
        "Kết quả": np.where(result["prediction"].eq(1), "Nghi gian lận", "Bình thường"),
    })
    if group != "Tất cả giao dịch":
        display_table = display_table.loc[display_table["Kết quả"].eq(group)]
    if sort_scores:
        display_table = display_table.sort_values("fraud_score", ascending=False, kind="stable")
    st.dataframe(
        display_table, hide_index=True, width="stretch",
        column_config={
            "fraud_score": st.column_config.ProgressColumn("Điểm dự đoán gian lận", min_value=0, max_value=1, format="%.4f"),
            "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
        },
    )
    st.caption(f"Đang hiển thị {len(display_table):,}/{total:,} giao dịch. Số dòng tính từ 1, không gồm tiêu đề CSV.")
    st.download_button(
        "Tải CSV kết quả", data=download_data,
        file_name=f"{Path(filename).stem}_predictions.csv", mime="text/csv", type="primary",
    )
    st.caption("CSV xuất gồm toàn bộ giao dịch theo thứ tự gốc và hai cột fraud_score, prediction; bộ lọc chỉ ảnh hưởng bảng hiển thị.")

    evaluation, label_message = evaluate_labels(result)
    if label_message:
        st.info(label_message)
    if evaluation is not None:
        st.subheader("Đánh giá với nhãn thực tế")
        st.caption("Kết quả trên dữ liệu vừa chọn/tải lên, không phải kết quả test chính thức của project.")
        for column, (label, value) in zip(st.columns(3), evaluation["metrics"].items()):
            column.metric(label, f"{value:.4f}", border=True)
        matrix = evaluation["matrix"]
        figure = go.Figure(go.Heatmap(
            z=matrix, x=["Bình thường (0)", "Gian lận (1)"], y=["Bình thường (0)", "Gian lận (1)"],
            text=matrix, texttemplate="%{text}", colorscale="Blues", showscale=False,
            hovertemplate="Thực tế: %{y}<br>Dự đoán: %{x}<br>Số giao dịch: %{z}<extra></extra>",
        ))
        figure.update_layout(
            title="Confusion Matrix", xaxis_title="Class dự đoán", yaxis_title="Class thực tế",
            yaxis=dict(autorange="reversed"), height=330, margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        if not flagged:
            st.caption("Không có giao dịch bị cảnh báo; Precision được quy ước bằng 0.")


def main():
    st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="🛡️", layout="wide")
    st.markdown("""
    <style>
    .block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem;}
    h1 {letter-spacing: -0.035em;}
    [data-testid="stMetricValue"] {font-size: 1.9rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #e3e9f2;}
    </style>
    """, unsafe_allow_html=True)
    try:
        model, threshold, features = load_artifacts()
    except Exception as error:
        st.title("Credit Card Fraud Detection")
        st.error(f"Không thể tải mô hình: {error}")
        st.stop()

    with st.sidebar:
        st.markdown("### 🛡️ Fraud Detection")
        st.caption("BÀI TẬP LỚN · HỌC MÁY")
        st.divider()
        st.markdown("**Mô hình**  \nRandom Forest")
        st.markdown("**Dataset**  \nCredit Card Fraud Detection (ULB)")
        st.metric("Ngưỡng phân loại", f"{threshold:.2f}")
        st.markdown("**Class 0** · Bình thường  \n**Class 1** · Nghi gian lận")
        st.divider()
        st.caption("30 đặc trưng đầu vào thô. Dữ liệu tải lên được xử lý trong bộ nhớ phiên làm việc, không ghi ra ổ đĩa.")

    st.caption("PHÂN TÍCH GIAO DỊCH · RANDOM FOREST")
    st.title("Credit Card Fraud Detection")
    st.write("Hệ thống phát hiện gian lận giao dịch thẻ tín dụng bằng Machine Learning.")
    st.divider()
    st.subheader("Dữ liệu giao dịch")
    source = st.radio("Chọn nguồn dữ liệu", ["Tải CSV", "Dùng dữ liệu demo"], horizontal=True)
    payload, filename = None, None
    if source == "Tải CSV":
        uploaded = st.file_uploader("Tải file giao dịch", type=["csv"], help="CSV UTF-8, phân cách bằng dấu phẩy. Cần Time, V1–V28, Amount; Class là tùy chọn.")
        if uploaded is not None:
            payload, filename = uploaded.getvalue(), uploaded.name
    else:
        if DEMO_PATH.is_file():
            payload, filename = DEMO_PATH.read_bytes(), DEMO_PATH.name
            st.info("Demo gồm 20 giao dịch thật được trích từ dataset gốc: 10 Class 0 và 10 Class 1. Mẫu minh họa này không đại diện tỷ lệ gian lận của dataset.")
        else:
            st.warning("Chưa có CSV demo. Bạn có thể tải CSV giao dịch của mình.")
    with st.expander("Định dạng CSV đầu vào"):
        st.write("Cần đủ 30 cột số bên dưới; có thể thêm Class hoặc cột ID. Không chuẩn hóa lại Time/Amount.")
        st.code(", ".join(features), language=None)
        if DEMO_PATH.is_file():
            st.download_button("Tải CSV demo", DEMO_PATH.read_bytes(), file_name=DEMO_PATH.name, mime="text/csv")

    source_key = hashlib.sha256(payload).hexdigest() if payload is not None else None
    if st.session_state.get("source_key") != (source_key, filename):
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_csv", None)
        st.session_state["source_key"] = (source_key, filename)
    run = st.button("Phân tích giao dịch", type="primary", disabled=payload is None)
    if payload is None:
        st.info("Tải CSV hoặc chọn dữ liệu demo để bắt đầu phân tích.")
    if run:
        try:
            with st.spinner("Đang kiểm tra dữ liệu và dự đoán…"):
                frame = read_transactions(payload, features)
                result = predict_transactions(frame, model, features, threshold)
                st.session_state["prediction_result"] = result
                st.session_state["prediction_csv"] = result.to_csv(index=False).encode("utf-8-sig")
        except Exception as error:
            st.session_state.pop("prediction_result", None)
            st.session_state.pop("prediction_csv", None)
            st.error(f"Không thể phân tích: {error}")
    if "prediction_result" in st.session_state:
        st.divider()
        render_results(st.session_state["prediction_result"], filename, threshold, st.session_state["prediction_csv"])


if __name__ == "__main__":
    main()
