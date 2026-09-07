import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. CẤU HÌNH TRANG WEB
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
        return None

model = load_model()

# ==========================================
# 3. GIAO DIỆN CHÍNH (Chia thành 2 Tabs)
# ==========================================
st.title("🛒 Trợ lý AI Phân tích Hành vi Khách hàng")
st.markdown("---")

tab1, tab2 = st.tabs(["🎯 Công cụ Dự đoán", "🧠 Phân tích Thuật toán LightGBM"])

# ---------------------------------------------------------
# TAB 1: CÔNG CỤ DỰ ĐOÁN (UI CHO NGƯỜI DÙNG)
# ---------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1])

    # --- BÊN TRÁI: NHẬP LIỆU ---
    with col1:
        st.subheader("📝 Nhập dữ liệu hành vi của khách")
        
        prod_duration = st.slider("⏱️ Thời gian xem Sản phẩm (giây)", 0, 5000, 1500)
        admin_duration = st.slider("⚙️ Thời gian ở trang Quản lý/Thanh toán", 0, 1000, 50)
        bounce_rate = st.slider("🏃 Tỷ lệ thoát nhanh (Bounce Rate)", 0.0, 0.2, 0.01)
        
        col_a, col_b = st.columns(2)
        with col_a:
            month = st.selectbox("Tháng truy cập", ["Feb", "Mar", "May", "Oct", "Nov", "Dec"], index=4)
            visitor_type = st.selectbox("Loại khách hàng", ["New_Visitor", "Returning_Visitor", "Other"])
        with col_b:
            weekend = st.radio("Truy cập vào cuối tuần?", [True, False])
            region = st.selectbox("Khu vực địa lý (Region ID)", [1, 2, 3, 4, 5, 6, 7, 8, 9])

    # --- BÊN PHẢI: KẾT QUẢ ---
    with col2:
        st.subheader("🤖 Kết quả Dự đoán từ AI")
        
        if st.button("🚀 Bấm để AI Chấm Điểm", use_container_width=True):
            if model is None:
                st.warning("Không có mô hình AI để chạy.")
            else:
                try:
                    # 1. Đóng gói ĐỦ 17 CỘT theo chuẩn bộ dữ liệu Kaggle
                    input_data = pd.DataFrame([{
                        "Administrative": 0,
                        "Administrative_Duration": float(admin_duration),
                        "Informational": 0,
                        "Informational_Duration": 0.0,
                        "ProductRelated": 10, 
                        "ProductRelated_Duration": float(prod_duration),
                        "BounceRates": float(bounce_rate),
                        "ExitRates": float(bounce_rate) + 0.01, # Giả lập tương quan thực tế
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
                    
                    # 2. Xử lý chuẩn xác các cột Categorical
                    cat_features = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
                    for col in cat_features:
                        input_data[col] = input_data[col].astype(str).astype("category")
                    
                    # 3. Ép chạy 1 luồng để chống treo hệ thống (Anti-freeze)
                    model.set_params(n_jobs=1)
                    
                    # 4. Dự đoán xác suất
                    prob = model.predict_proba(input_data)[0][1]
                    
                    # 5. Hiển thị UI kết quả
                    st.write("### Xác suất chốt đơn:")
                    st.progress(float(prob))
                    st.metric(label="", value=f"{prob * 100:.2f}%")
                    
                    st.markdown("### 💡 Đề xuất hành động Marketing:")
                    if prob >= 0.7:
                        st.success("**🟢 KHÁCH HÀNG RẤT NÉT!** \n\n Không cần mã giảm giá, cứ để họ tự chốt đơn (Tránh lãng phí lợi nhuận).")
                    elif prob >= 0.4:
                        st.warning("**🟠 KHÁCH ĐANG PHÂN VÂN!** \n\n Nên hiển thị Popup tặng mã giảm giá 10% (hoặc Freeship) để chốt sale tức thì.")
                    else:
                        st.error("**🔴 KHÁCH VÃNG LAI.** \n\n Xác suất chốt đơn quá thấp. Không nên lưu Data để chạy Retargeting Ads cho người này.")
                        
                except Exception as e:
                    st.error(f"❌ Thuật toán gặp lỗi khi dự đoán: {e}")

# ---------------------------------------------------------
# TAB 2: MINH HỌA THUẬT TOÁN (TÍNH NĂNG MỚI)
# ---------------------------------------------------------
with tab2:
    st.subheader("🕵️ Trực quan hóa Cây quyết định của LightGBM")
    st.markdown("""
    Thuật toán **Light Gradient Boosting Machine (LightGBM)** đưa ra quyết định không phải dựa trên cảm tính, 
    mà dựa trên việc học hỏi từ hàng vạn khách hàng trong quá khứ. Dưới đây là những yếu tố mà mô hình cho là quan trọng nhất khi đánh giá một khách hàng.
    """)
    
    if model is not None:
        try:
            # Trích xuất tầm quan trọng của các đặc trưng (Feature Importance)
            importance_values = model.feature_importances_
            feature_names = model.feature_name_
            
            # Tạo DataFrame và sắp xếp
            df_importance = pd.DataFrame({
                'Mức độ đóng góp': importance_values
            }, index=feature_names)
            df_importance = df_importance.sort_values(by='Mức độ đóng góp', ascending=False)
            
            # Vẽ biểu đồ thanh ngang
            st.bar_chart(df_importance)
            
            st.info("""
            **💡 Diễn giải biểu đồ:** 
            - Các cột càng cao thể hiện yếu tố đó càng chi phối mạnh mẽ đến quyết định mua hàng của khách.
            - Nếu **PageValues** (Giá trị trang) hoặc **ProductRelated_Duration** (Thời gian xem sản phẩm) nằm trên top, điều đó chứng tỏ việc giữ chân khách hàng đọc kỹ thông tin sản phẩm mang lại tỷ lệ chuyển đổi cao nhất.
            """)
        except Exception as e:
            st.error(f"Không thể vẽ biểu đồ Feature Importance: {e}")
    else:
        st.warning("Vui lòng tải mô hình thành công để xem phân tích.")
