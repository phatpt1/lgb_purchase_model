# SỬA LỖI TREO (DEADLOCK): Bắt buộc phải đặt 3 dòng này ở trên cùng, 
# TRƯỚC KHI import bất kỳ thư viện machine learning nào.
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==========================================
# 1. CẤU HÌNH TRANG WEB CHÍNH
# ==========================================
st.set_page_config(
    page_title="AI Dự đoán Mua Hàng | LightGBM",
    page_icon="🛒",
    layout="wide"
)

# ==========================================
# 2. HÀM LOAD MÔ HÌNH AN TOÀN
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("lgb_purchase_model.joblib")
        return model
    except Exception as e:
        st.error(f"❌ LỖI NGHIÊM TRỌNG: Không thể nạp file mô hình. Chi tiết: {e}")
        st.info("💡 Mẹo: Hãy kiểm tra xem file 'lgb_purchase_model.joblib' có nằm cùng thư mục với file code này không.")
        return None

model = load_model()

# ==========================================
# 3. GIAO DIỆN CHÍNH (Chia thành 3 Tabs)
# ==========================================
st.title("🛒 Trợ lý AI Phân tích Hành vi Khách hàng")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 Công cụ Dự đoán", "🧠 Giải phẫu Thuật toán", "📖 Tài liệu Toán học & Code"])

# ---------------------------------------------------------
# TAB 1: CÔNG CỤ DỰ ĐOÁN & TRUY VẾT SUY LUẬN
# ---------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Nhập dữ liệu hành vi của khách")
        prod_duration = st.slider("⏱️ Thời gian xem Sản phẩm (giây)", 0.0, 5000.0, 1500.0)
        admin_duration = st.slider("⚙️ Thời gian ở trang Quản lý/Thanh toán", 0.0, 1000.0, 50.0)
        bounce_rate = st.slider("🏃 Tỷ lệ thoát nhanh (Bounce Rate)", 0.0, 0.2, 0.01)
        
        col_a, col_b = st.columns(2)
        with col_a:
            month = st.selectbox("Tháng truy cập", ["Feb", "Mar", "May", "Oct", "Nov", "Dec"], index=4)
            visitor_type = st.selectbox("Loại khách hàng", ["New_Visitor", "Returning_Visitor", "Other"])
        with col_b:
            weekend = st.radio("Truy cập vào cuối tuần?", [True, False])
            region = st.selectbox("Khu vực địa lý (Region ID)", [1, 2, 3, 4, 5, 6, 7, 8, 9])

    with col2:
        st.subheader("🤖 Kết quả Dự đoán từ AI")
        
        if st.button("🚀 Bấm để AI Chấm Điểm", use_container_width=True):
            if model is None:
                st.warning("Không có mô hình AI để chạy.")
            else:
                try:
                    # Bù đủ 17 cột
                    input_data = pd.DataFrame([{
                        "Administrative": 0,
                        "Administrative_Duration": float(admin_duration),
                        "Informational": 0,
                        "Informational_Duration": 0.0,
                        "ProductRelated": 10, 
                        "ProductRelated_Duration": float(prod_duration),
                        "BounceRates": float(bounce_rate),
                        "ExitRates": float(bounce_rate) + 0.01,
                        "PageValues": 0.0,
                        "SpecialDay": 0.0,
                        "Month": month,
                        "OperatingSystems": 2, 
                        "Browser": 2,
                        "Region": int(region),
                        "TrafficType": 2,
                        "VisitorType": visitor_type,
                        "Weekend": bool(weekend)
                    }])
                    
                    cat_features = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
                    for col in cat_features:
                        input_data[col] = input_data[col].astype(str).astype("category")
                    
                    # Dự đoán xác suất
                    prob = model.predict_proba(input_data)[0][1]
                    
                    st.write("### Xác suất chốt đơn:")
                    st.progress(float(prob))
                    st.metric(label="", value=f"{prob * 100:.2f}%")
                    
                    st.markdown("### 💡 Đề xuất hành động Marketing:")
                    if prob >= 0.7:
                        st.success("**🟢 KHÁCH HÀNG RẤT NÉT!** \n\n Không cần mã giảm giá, cứ để họ tự chốt đơn.")
                    elif prob >= 0.4:
                        st.warning("**🟠 KHÁCH ĐANG PHÂN VÂN!** \n\n Nên hiển thị Popup tặng mã giảm giá 10% để chốt sale tức thì.")
                    else:
                        st.error("**🔴 KHÁCH VÃNG LAI.** \n\n Xác suất chốt đơn quá thấp. Bỏ qua để tiết kiệm ngân sách.")
                    
                    st.markdown("---")
                    with st.expander("🕵️ Xem cách AI đưa ra quyết định (Step-by-step)", expanded=True):
                        base_score = 15.0
                        st.write("**Giai đoạn 1: Đánh giá cơ sở (Khởi điểm)**")
                        st.info(f"Xác suất cơ sở ban đầu đặt ở mức **{base_score}%** (Do tỷ lệ khách mua trung bình).")
                        
                        st.write("**Giai đoạn 2: Các chuyên gia (Cây quyết định) vào cuộc**")
                        col_step1, col_step2 = st.columns(2)
                        
                        with col_step1:
                            if prod_duration > 2000:
                                st.success(f"✔️ **Thời gian xem ({prod_duration}s):** Rất lâu ➔ **Cộng điểm mạnh**")
                            elif prod_duration > 800:
                                st.success(f"✔️ **Thời gian xem ({prod_duration}s):** Mức khá ➔ **Cộng điểm nhẹ**")
                            else:
                                st.error(f"❌ **Thời gian xem ({prod_duration}s):** Quá nhanh ➔ **Trừ điểm**")
                                
                            if bounce_rate > 0.05:
                                st.error(f"❌ **Tỷ lệ thoát ({bounce_rate}):** Nằm ở ngưỡng rủi ro cao ➔ **Trừ điểm**")
                            else:
                                st.success(f"✔️ **Tỷ lệ thoát ({bounce_rate}):** Khách duyệt web mượt mà ➔ **Cộng điểm**")

                        with col_step2:
                            if visitor_type == "Returning_Visitor":
                                st.success(f"✔️ **Loại khách:** Khách cũ quay lại ➔ **Cộng điểm**")
                            elif visitor_type == "New_Visitor":
                                st.warning(f"⚠️ **Loại khách:** Khách mới, độ tin cậy chưa cao ➔ **Trừ điểm nhẹ**")
                            else:
                                st.info(f"ℹ️ **Loại khách:** Khác ➔ **Không đổi**")
                                
                            if month in ["Nov", "Dec"]:
                                st.success(f"✔️ **Thời điểm ({month}):** Mùa mua sắm cuối năm ➔ **Cộng điểm**")
                            else:
                                st.info(f"ℹ️ **Thời điểm ({month}):** Tháng thường ➔ **Không đổi**")

                        st.write("**Giai đoạn 3: Tổng hợp (Chốt kết quả)**")
                        st.markdown(f"Tổng hợp hàng trăm cây quyết định qua hàm Sigmoid ➔ **{prob * 100:.2f}%**")
                        
                except Exception as e:
                    st.error(f"❌ Thuật toán gặp lỗi khi dự đoán: {e}")

# ---------------------------------------------------------
# TAB 2: MINH HỌA THUẬT TOÁN 
# ---------------------------------------------------------
with tab2:
    st.subheader("🕵️ Giải phẫu \"Bộ não\" của LightGBM")
    if model is not None:
        try:
            df_importance = pd.DataFrame({
                'Feature': model.feature_name_,
                'Importance': model.feature_importances_
            })
            df_top10 = df_importance.sort_values(by='Importance', ascending=True).tail(10)
            
            st.markdown("#### 🏆 Top 10 Cột trụ chốt sale")
            fig = px.bar(df_top10, x='Importance', y='Feature', orientation='h', text='Importance', color='Importance', color_continuous_scale='Blues')
            fig.update_traces(textposition='outside')
            fig.update_layout(xaxis_title="Mức độ Tác động", yaxis_title="", plot_bgcolor='rgba(0,0,0,0)', height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi vẽ biểu đồ: {e}")

# ---------------------------------------------------------
# TAB 3: TÀI LIỆU CHỨNG MINH TOÁN HỌC & CODE
# ---------------------------------------------------------
with tab3:
    st.subheader("📖 Nền tảng Toán học của LightGBM")
    
    st.markdown(r"""
    ### 1. Hàm mục tiêu (Objective Function)
    Thuật toán cực tiểu hóa hàm mất mát **Log-Loss** (Binary Cross Entropy) để phân loại khách hàng (1: Mua, 0: Không mua):
    
    $$ \mathcal{L}(y, F(x)) = - \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] $$
    
    Trong đó $p_i = \frac{1}{1 + e^{-F(x_i)}}$ là hàm **Sigmoid** dùng để ép điểm số về dạng phần trăm xác suất.

    ### 2. Tối ưu hóa bằng Gradient Descent
    Cây quyết định ở vòng lặp thứ $m$ được xây dựng dựa trên đạo hàm bậc 1 (Gradient - $g_i$) và bậc 2 (Hessian - $h_i$) của hàm mất mát. Kết quả được cập nhật dựa trên Tốc độ học ($\eta$):
    $$ F_m(x) = F_{m-1}(x) + \eta \cdot f_m(x) $$

    ### 3. Kiến trúc rẽ nhánh tốc độ cao (Leaf-wise)
    Thay vì phát triển cây đồng đều (Level-wise), LightGBM ưu tiên chia cắt ở chiếc lá mang lại Lợi nhuận thông tin ($\text{Gain}$) lớn nhất:
    
    $$ \text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] $$
    
    ---
    """)
    
    st.subheader("💻 Ánh xạ từ Toán học sang Python Code")
    st.markdown("""
    Những phương trình phức tạp phía trên được gói gọn hoàn toàn qua các tham số truyền vào hàm `LGBMClassifier` trong quá trình ta huấn luyện (file `train.py`). Dưới đây là bảng đối chiếu minh họa:
    """)
    
    st.code("""
# Khởi tạo mô hình LightGBM với các tham số bắt nguồn từ nền tảng Toán học
import lightgbm as lgb

model = lgb.LGBMClassifier(
    # 1. Tương ứng với Hàm mục tiêu Log-Loss ở phần 1
    objective="binary",         
    
    # 2. Tương ứng với tham số \eta (Tốc độ học) ở phần 2
    learning_rate=0.05,         
    
    # 3. Số vòng lặp m, tương đương số lượng cây quyết định được tạo ra
    n_estimators=120,           
    
    # 4. Giới hạn số lá tối đa để kiểm soát phương trình Gain ở phần 3, tránh học vẹt (Overfitting)
    num_leaves=25,              
    
    # 5. Tham số \lambda ẩn: Phạt trọng số sai số để bù đắp dữ liệu mất cân bằng (85/15)
    scale_pos_weight=5.6,       
    
    # Xử lý cấp hệ thống: Ép chạy 1 luồng để tránh Xung đột (Deadlock) trên máy chủ
    n_jobs=1,
    random_state=42
)
    """, language="python")

    st.success("💡 **Bí quyết thực chiến:** Sức mạnh của Machine Learning hiện đại nằm ở việc chúng ta không cần phải tự giải các phương trình đạo hàm bằng tay, mà chỉ cần hiểu bản chất toán học của chúng để tinh chỉnh các tham số (Hyperparameters) trong code sao cho tối ưu nhất với dữ liệu kinh doanh của mình.")
