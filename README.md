# DuLieuLon
link data: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
tải 2 file dữ liệu(Fake.csv, Real.csv)

## Mở terminal chạy "pip install -r requirements.txt"

Đảm bảo Hadoop/HDFS đang chạy (nếu dùng HDFS):

start-dfs.sh
start-yarn.sh
jps  

## Mở file Test1.inbpy

Chạy lần lượt các Cells (Cells) để thực hiện quy trình:

Load dữ liệu Fake.csv và True.csv.

Làm sạch (Data Cleaning): Xóa Null, xóa trùng lặp.

Tiền xử lý (Preprocessing): regexTokenizer, remover, ngram, hashingTF, idf, lr

Huấn luyện (Training): Logistic Regression.

Lưu Model: Code sẽ lưu model vào hdfs://localhost:9000/user/hdoop/fake_news_model_final

## Mở file app.py
Tại terminal (đã kích hoạt .venv), chạy lệnh:
streamlit run app.py
