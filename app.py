import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, regexp_replace, trim
import os
import sys

# --- CẤU HÌNH HỆ THỐNG ---
# Giúp Streamlit tìm thấy Python & PySpark trong môi trường ảo
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# --- 1. KHỞI TẠO SPARK & LOAD MÔ HÌNH TỪ HDFS ---
@st.cache_resource
def load_spark_model():
    # Khởi tạo Spark Session
    # master("local[*]"): Dùng tất cả nhân CPU của máy để chạy cho nhanh
    spark = SparkSession.builder \
        .appName("FakeNewsApp") \
        .master("local[*]") \
        .getOrCreate()
    
    # Load mô hình từ HDFS (Đã sửa đường dẫn thành localhost)
    model_path = "hdfs://localhost:9000/user/hdoop/fake_news_model_final"
    model = PipelineModel.load(model_path)
    return spark, model

# --- 2. GIAO DIỆN NGƯỜI DÙNG (UI) ---
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="centered")

st.title("🕵️ Phát hiện Tin giả (Fake News)")
st.caption("Hệ thống sử dụng **Apache Spark** & **Logistic Regression**")

# Tải mô hình (Chỉ chạy 1 lần đầu tiên)
try:
    with st.spinner('Đang kết nối HDFS và tải mô hình...'):
        spark, model = load_spark_model()
    st.success("✅ Hệ thống đã sẵn sàng!", icon="🟢")
except Exception as e:
    st.error(f"⚠️ Lỗi kết nối Spark/HDFS: {e}")
    st.info("Gợi ý: Hãy kiểm tra xem Hadoop đã bật chưa (lệnh `jps`)?")
    st.stop()

# Khung nhập liệu
user_input = st.text_area("Nhập nội dung tin tức tiếng Anh:", height=200, 
                          placeholder="Paste bài báo vào đây (Ví dụ: WASHINGTON (Reuters) - ...)")

# --- 3. XỬ LÝ DỰ ĐOÁN ---
if st.button("🔍 Kiểm tra độ tin cậy", type="primary"):
    if not user_input.strip():
        st.warning("Vui lòng nhập nội dung để kiểm tra.")
    else:
        with st.spinner('AI đang phân tích văn phong và từ vựng...'):
            # A. TẠO DATAFRAME TỪ INPUT
            df_test = spark.createDataFrame([(user_input,)], ["text"])
            
            # B. TIỀN XỬ LÝ THỦ CÔNG (Bắt buộc phải có bước này!)
            # Lý do: Pipeline chỉ xử lý dữ liệu sạch. Ta phải xóa rác (Dateline) trước.
            robust_pattern = r"^.*?\s*\(.*?\)\s*-\s*" # Mẫu xóa: "WASHINGTON (Reuters) - "
            
            df_clean = df_test.withColumn("text", regexp_replace(col("text"), robust_pattern, ""))
            df_clean = df_clean.withColumn("text", trim(col("text")))
            
            # C. DỰ ĐOÁN (Chạy qua Pipeline: Tokenizer -> Remover -> TF -> IDF -> Model)
            prediction = model.transform(df_clean)
            
            # D. LẤY KẾT QUẢ
            result = prediction.select("prediction", "probability").collect()[0]
            is_fake = (result['prediction'] == 0.0) # 0.0 là Fake (theo nhãn của tập Fake.csv)
            probs = result['probability']
            
            # E. HIỂN THỊ KẾT QUẢ
            st.divider()
            
            if is_fake:
                # Trường hợp Tin Giả
                confidence = probs[0] * 100
                st.error(f"🚨 KẾT QUẢ: TIN GIẢ (FAKE NEWS)")
                st.metric(label="Độ tin cậy của dự đoán", value=f"{confidence:.2f}%")
                st.warning("Cảnh báo: Bài viết này có văn phong giật gân, thiếu cấu trúc chuẩn của báo chí.")
            else:
                # Trường hợp Tin Thật
                confidence = probs[1] * 100
                st.success(f"✅ KẾT QUẢ: TIN THẬT (REAL NEWS)")
                st.metric(label="Độ tin cậy của dự đoán", value=f"{confidence:.2f}%")
                st.info("Bài viết này có cấu trúc và từ vựng phù hợp với tin tức chính thống.")

# --- Footer ---
st.markdown("---")
st.markdown("*Demo Project - Big Data with PySpark*")