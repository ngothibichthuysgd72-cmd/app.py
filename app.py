import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import io

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION (Lệnh Streamlit đầu tiên)
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Gian lận tại Agribank",
    page_icon="👍"
)

# -----------------------------------------------------------------------------
# 2. CACHED FUNCTIONS & UTILITIES
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Đang tải và xử lý dữ liệu...")
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes để tối ưu cache và tránh xung đột trạng thái."""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# -----------------------------------------------------------------------------
# 3. SIDEBAR - VÙNG CẤU HÌNH & ĐIỀU KHIỂN
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình & Dữ liệu")
    
    # 3.1 Tải dữ liệu huấn luyện mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện mẫu (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Tải tập dữ liệu gốc ban đầu chứa biến mục tiêu 'default' để huấn luyện mô hình."
    )
    
    st.divider()
    
    # 3.2 Lựa chọn mô hình thuật toán dựa trên Notebook
    st.subheader("🤖 Thuật toán & Siêu tham số")
    model_choice = st.selectbox(
        "Chọn mô hình huấn luyện:",
        options=["Random Forest", "Decision Tree", "Logistic Regression"],
        index=0,
        help="Hệ thống cung cấp 3 mô hình tương ứng trong tiến trình thực nghiệm của bạn."
    )
    
    # Thiết lập siêu tham số động theo lựa chọn mô hình dựa trên notebook gốc
    hyperparams = {}
    hyperparams['random_state'] = st.number_input(
        "Random State", 
        value=32, 
        step=1, 
        help="Giá trị thiết lập nhằm cố định tính ngẫu nhiên khi phân tách và huấn luyện."
    )
    
    if model_choice == "Random Forest":
        hyperparams['n_estimators'] = st.slider(
            "Số lượng cây quyết định (n_estimators)", 
            min_value=10, 
            max_value=200, 
            value=10, # Mặc định theo mã khởi tạo tối ưu
            step=10,
            help="Số lượng cây quyết định trong phân nhóm rừng ngẫu nhiên."
        )
    elif model_choice == "Decision Tree":
        hyperparams['criterion'] = st.selectbox(
            "Tiêu chí phân tách (criterion)", 
            options=["gini", "entropy", "log_loss"],
            help="Hàm đo lường chất lượng phân tách của các nút thuộc cây."
        )
    elif model_choice == "Logistic Regression":
        hyperparams['max_iter'] = st.number_input(
            "Số vòng lặp tối đa (max_iter)", 
            value=1000, 
            step=100,
            help="Số vòng lặp tối đa cho các bộ tối ưu hội tụ."
        )
        
    st.divider()
    
    # 3.3 Nút kích hoạt tác vụ Huấn luyện duy nhất
    train_clicked = st.button(
        "🚀 Kích hoạt Huấn luyện Mô hình", 
        type="primary", 
        use_container_width=True,
        help="Nhấn để bắt đầu trích xuất biến, chia tập dữ liệu huấn luyện/kiểm thử và khớp mô hình."
    )

# -----------------------------------------------------------------------------
# 4. HEADER - VÙNG ĐỊNH HƯỚNG ỨNG DỤNG
# -----------------------------------------------------------------------------
st.title("😍😍 Hệ thống và máy Phát hiện Gian lận tại Agribank")
st.caption("Ứng dụng quản trị thông minh hỗ trợ phân tách rủi ro tín dụng và tự động phát hiện hành vi mặc định (default) từ dữ liệu thực nghiệm.")

# Trình chặn trạng thái rỗng - Kiểm tra dữ liệu đầu vào
if uploaded_file is None:
    st.info("💡 Vui lòng tiến hành tải tệp dữ liệu mẫu (.csv hoặc .xlsx) tại Sidebar bên trái để bắt đầu kích hoạt hệ thống.")
    st.stop()

# Đọc dữ liệu khi đã có file tải lên thành công
file_bytes = uploaded_file.getvalue()
df = load_data(file_bytes, uploaded_file.name)

if df is None:
    st.error("Không thể xử lý định dạng tệp đã chọn. Vui lòng kiểm tra lại cấu trúc.")
    st.stop()

st.caption(f"📁 Đang ghi nhận tệp nguồn dữ liệu: **{uploaded_file.name}**")
st.divider()

# -----------------------------------------------------------------------------
# 5. KHỐI LOGIC HUẤN LUYỆN (Chỉ kích hoạt khi bấm nút, lưu kết quả qua Session State)
# -----------------------------------------------------------------------------
TARGET_COL = 'default'

if train_clicked:
    if TARGET_COL not in df.columns:
        st.sidebar.error(f"❌ Không tìm thấy biến mục tiêu '{TARGET_COL}' trong tập dữ liệu.")
    else:
        with st.spinner("Đang xây dựng Pipeline xử lý và huấn luyện mô hình học máy..."):
            # Trích xuất tập biến đặc trưng X và mục tiêu y theo thiết lập notebook
            y = df[TARGET_COL]
            X = df.drop(columns=[TARGET_COL])
            
            # Phân tách tập dữ liệu Train/Test (Tỷ lệ 80/20, random_state cố định theo cấu hình)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=hyperparams['random_state']
            )
            
            # Khởi tạo mô hình thuật toán theo lựa chọn
            if model_choice == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=hyperparams['n_estimators'], 
                    random_state=hyperparams['random_state']
                )
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier(
                    criterion=hyperparams['criterion'], 
                    random_state=hyperparams['random_state']
                )
            else:
                model = LogisticRegression(
                    max_iter=hyperparams['max_iter'], 
                    random_state=hyperparams['random_state']
                )
                
            # Khớp mô hình
            model.fit(X_train, y_train)
            
            # Dự báo kiểm định thử trên tập Test
            y_pred = model.predict(X_test)
            
            # Lưu trữ toàn bộ kết quả vào session_state để tái sử dụng xuyên suốt các tab
            st.session_state['trained_model'] = model
            st.session_state['model_name'] = model_choice
            st.session_state['features_list'] = X.columns.tolist()
            st.session_state['feature_types'] = X.dtypes.to_dict()
            st.session_state['X_train_summary'] = X_train.describe()
            
            # Lưu chỉ số đo lường hiệu năng
            st.session_state['metrics'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'confusion': confusion_matrix(y_test, y_pred),
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
            st.sidebar.success(f"✔️ Huấn luyện xong mô hình {model_choice}!")

# -----------------------------------------------------------------------------
# 6. GIAO DIỆN TABS CHỨC NĂNG CHÍNH
# -----------------------------------------------------------------------------
tab_overview, tab_viz, tab_eval, tab_predict = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🧪 Kết quả kiểm định mô hình", 
    "🔮 Sử dụng mô hình dự báo"
])

# --- TAB 1: TỔNG QUAN DỮ LIỆU ---
with tab_overview:
    st.subheader("📋 Cấu trúc và Thuộc tính thống kê")
    
    col_shape1, col_shape2, col_shape3 = st.columns(3)
    with col_shape1:
        st.metric("Tổng số dòng dữ liệu (Rows)", f"{df.shape[0]:,}")
    with col_shape2:
        st.metric("Tổng số lượng cột (Columns)", f"{df.shape[1]}")
    with col_shape3:
        file_size_mb = len(file_bytes) / (1024 * 1024)
        st.metric("Dung lượng tệp nguồn", f"{file_size_mb:.2f} MB")
        
    st.markdown("### 🔍 Xem dữ liệu thô (5 dòng đầu tiên)")
    st.dataframe(df.head(5), use_container_width=True)
    
    st.markdown("### 📈 Thống kê mô tả các thuộc tính đặc trưng (X & y)")
    # Chỉ mô tả các cột có sẵn trong tập dữ liệu tương tác
    st.dataframe(df.describe().T, use_container_width=True)

# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tab_viz:
    st.subheader("📊 Trực quan hóa phân phối biến thuộc tính")
    
    # Ưu tiên hiển thị biến mục tiêu đầu tiên nếu có trong danh sách
    if TARGET_COL in df.columns:
        fig_target = px.histogram(
            df, x=TARGET_COL, 
            title="Phân phối trạng thái Rủi ro / Gian lận (Biến mục tiêu: default)",
            color=TARGET_COL,
            text_auto=True,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_target.update_layout(height=350, bargap=0.3)
        st.plotly_chart(fig_target, use_container_width=True)
    
    # Trực quan các đặc trưng X (Mặc định chọn tối đa 4 thuộc tính đầu tiên để vẽ phân phối lưới 2x2)
    feature_cols = [col for col in df.columns if col != TARGET_COL]
    
    if len(feature_cols) > 0:
        st.markdown("---")
        st.markdown("### 🛠️ Phân tích phân phối các biến đầu vào")
        selected_viz_cols = st.multiselect(
            "Chọn các biến đầu vào cần trực quan hóa:",
            options=feature_cols,
            default=feature_cols[:min(4, len(feature_cols))],
            help="Giới hạn hoặc mở rộng lựa chọn để phân tách cấu trúc đồ thị."
        )
        
        if selected_viz_cols:
            # Tạo lưới đồ thị động dựa trên số lượng cột người dùng lựa chọn
            for i in range(0, len(selected_viz_cols), 2):
                cols_grid = st.columns(2)
                for j in range(2):
                    idx = i + j
                    if idx < len(selected_viz_cols):
                        col_name = selected_viz_cols[idx]
                        with cols_grid[j]:
                            # Xác định kiểu dữ liệu để chọn biểu đồ phù hợp
                            if pd.api.types.is_numeric_dtype(df[col_name]):
                                fig_feat = px.histogram(
                                    df, x=col_name, 
                                    title=f"Phân phối số của biến: {col_name}",
                                    marginal="box",
                                    color_discrete_sequence=['#1f77b4']
                                )
                            else:
                                fig_feat = px.bar(
                                    df[col_name].value_counts().reset_index(),
                                    x='index', y=col_name,
                                    title=f"Tần suất phân loại của biến: {col_name}",
                                    text_auto=True
                                )
                            fig_feat.update_layout(height=300)
                            st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.warning("Tập dữ liệu không chứa biến đầu vào hợp lệ.")

# --- TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH ---
with tab_eval:
    st.subheader("🧪 Đánh giá độ tin cậy và chất lượng mô hình")
    
    if 'trained_model' not in st.session_state:
        st.info("ℹ️ Mô hình chưa được kích hoạt huấn luyện. Vui lòng chọn tham số và nhấn nút 'Kích hoạt Huấn luyện Mô hình' ở Sidebar bên trái.")
    else:
        metrics = st.session_state['metrics']
        model_name = st.session_state['model_name']
        
        st.success(f"🎉 Hiển thị báo cáo đánh giá của mô hình hiện tại: **{model_name}**")
        
        # 4 Chỉ số đánh giá cốt lõi (Phân loại nhị phân)
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Độ chính xác tổng thể (Accuracy)", f"{metrics['accuracy']:.4f}")
        with m_col2:
            st.metric("Độ chính xác mô hình dự báo (Precision)", f"{metrics['precision']:.4f}")
        with m_col3:
            st.metric("Tỷ lệ bắt sót gian lận (Recall)", f"{metrics['recall']:.4f}")
        with m_col4:
            st.metric("Chỉ số cân bằng F1-Score", f"{metrics['f1']:.4f}")
            
        st.markdown("---")
        
        eval_col1, eval_col2 = st.columns(2)
        
        with eval_col1:
            st.markdown("### 🧩 Ma trận nhầm lẫn (Confusion Matrix)")
            cm = metrics['confusion']
            # Vẽ trực quan ma trận nhầm lẫn bằng biểu đồ nhiệt Heatmap của Plotly
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Nhãn Dự Đoán (Predicted)", y="Nhãn Thực Tế (Actual)", color="Số lượng"),
                x=['Không gian lận (0)', 'Gian lận/Nợ xấu (1)'],
                y=['Không gian lận (0)', 'Gian lận/Nợ xấu (1)'],
                text_auto=True,
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(height=350)
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with eval_col2:
            st.markdown("### 📋 Classification Report Chi Tiết")
            report_df = pd.DataFrame(metrics['report']).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)

# --- TAB 4: SỬ DỤNG MÔ HÌNH DỰ BÁO ---
with tab_predict:
    st.subheader("🔮 Triển khai nhận diện và chấm điểm rủi ro trực tuyến")
    
    if 'trained_model' not in st.session_state:
        st.info("ℹ️ Vui lòng huấn luyện mô hình ở Sidebar trước khi thực hiện các dự báo đơn lẻ hoặc hàng loạt.")
    else:
        model = st.session_state['trained_model']
        features = st.session_state['features_list']
        X_train_summary = st.session_state['X_train_summary']
        
        predict_mode = st.radio(
            "Chọn phương thức nhập liệu đầu vào:",
            options=["Nhập trực tiếp giá trị đơn lẻ", "Tải tệp danh sách cần chấm điểm hàng loạt"],
            horizontal=True
        )
        
        # CHẾ ĐỘ 1: NHẬP TRỰC TIẾP QUA FORM
        if predict_mode == "Nhập trực tiếp giá trị đơn lẻ":
            st.markdown("### 📝 Điền thông số chi tiết của giao dịch / khách hàng")
            
            with st.form("single_prediction_form"):
                # Tạo form nhập liệu động tự động khớp theo số lượng đặc trưng từ Notebook
                form_cols = st.columns(3)
                input_data = {}
                
                for idx, col_name in enumerate(features):
                    col_pos = idx % 3
                    with form_cols[col_pos]:
                        # Thiết lập giá trị mặc định dựa vào trung vị hoặc trung bình của tập huấn luyện mẫu
                        default_val = float(X_train_summary.loc['50%', col_name]) if col_name in X_train_summary.columns else 0.0
                        min_val = float(X_train_summary.loc['min/', col_name]) if col_name in X_train_summary.columns else 0.0
                        max_val = float(X_train_summary.loc['max', col_name]) if col_name in X_train_summary.columns else 100.0
                        
                        input_data[col_name] = st.number_input(
                            f"Biến đặc trưng: {col_name}",
                            value=default_val,
                            help=f"Khoảng giá trị huấn luyện gốc từ {min_val} đến {max_val}"
                        )
                
                submit_pred = st.form_submit_button("🔍 Tiến hành phân tích rủi ro", type="primary", use_container_width=True)
                
            if submit_pred:
                # Chuyển đổi dữ liệu nhập vào thành DataFrame chuẩn hóa cấu trúc
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                
                st.markdown("### 🔔 Kết quả kiểm định độc lập:")
                if prediction == 1:
                    st.error("🚨 **CẢNH BÁO:** Hệ thống phát hiện đối tượng/giao dịch thuộc nhóm **RỦI RO GIAN LẬN / CÓ NGUY CƠ MẤT KHẢ NĂNG THANH TOÁN (default = 1)**.")
                else:
                    st.success("✅ **AN TOÀN:** Giao dịch/Đối tượng được phân loại thuộc nhóm **AN TOÀN / KHÔNG GIAN LẬN (default = 0)**.")
                    
                # Tính xác suất nếu mô hình hỗ trợ học xác suất đầu ra
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)[0]
                    st.info(f"📊 Độ tin cậy dự báo: Khả năng an toàn {probabilities[0]*100:.2f}% | Khả năng gian lận: {probabilities[1]*100:.2f}%")
                    
        # CHẾ ĐỘ 2: TẢI FILE DỰ BÁO HÀNG LOẠT (Tương thích định dạng X_new)
        else:
            st.markdown("### 📂 Nạp danh sách tập dữ liệu mới cần chấm điểm rủi ro")
            st.caption("Yêu cầu định dạng: Tệp phải chứa đầy đủ danh sách các cột đặc trưng đầu vào giống cấu trúc huấn luyện mẫu ban đầu.")
            
            batch_file = st.file_uploader(
                "Tải lên tệp dữ liệu kiểm toán mới (.csv, .xlsx)", 
                type=["csv", "xlsx"],
                key="batch_uploader"
            )
            
            if batch_file is not None:
                batch_bytes = batch_file.getvalue()
                if batch_file.name.endswith('.csv'):
                    new_df = pd.read_csv(io.BytesIO(batch_bytes))
                else:
                    new_df = pd.read_excel(io.BytesIO(batch_bytes))
                
                # Kiểm tra kiểm định Schema dữ liệu đầu vào
                missing_cols = [col for col in features if col not in new_df.columns]
                
                if missing_cols:
                    st.error(f"❌ Tệp tải lên thiếu các cột đặc trưng bắt buộc sau: {missing_cols}")
                else:
                    # Đảm bảo thứ tự sắp xếp cột đồng nhất với cấu trúc mô hình đã fit ban đầu
                    pred_features_df = new_df[features]
                    
                    with st.spinner("Hệ thống đang thực hiện quét dữ liệu và chấm điểm hàng loạt..."):
                        batch_preds = model.predict(pred_features_df)
                        
                        # Chèn cột kết quả dự đoán vào đầu bảng đầu ra
                        output_df = new_df.copy()
                        output_df.insert(0, 'Dự_Đoán_Rủi_Rao_default', batch_preds)
                        
                        # Hiển thị số lượng phát hiện
                        total_fraud = int(np.sum(batch_preds))
                        st.markdown("### 📊 Thống kê kết quả phân tích hàng loạt:")
                        bc1, bc2 = st.columns(2)
                        with bc1:
                            st.metric("Tổng số lượng trường hợp quét", f"{len(output_df):,}")
                        with bc2:
                            st.metric("Số trường hợp nghi vấn Gian lận / Nợ xấu", f"{total_fraud:,}", delta=f"{total_fraud/len(output_df)*100:.2f}% tỷ lệ rủi ro", delta_color="inverse")
                        
                        st.markdown("#### 🗒️ Bảng chi tiết kết quả tích hợp:")
                        st.dataframe(output_df, use_container_width=True)
                        
                        # Tạo nút xuất dữ liệu kết quả phân tích ra CSV
                        csv_buffer = io.StringIO()
                        output_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_output = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Tải xuống kết quả phân tích hệ thống (.CSV)",
                            data=csv_output,
                            file_name="Ket_Qua_Phan_Tich_Rui_Ro_Gian_Lan.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
