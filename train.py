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

# Gọi hàm load model khi mở trang
model = load_model()

# ==========================================
# 3. GIAO DIỆN CHÍNH (Chia thành 2 Tabs)
# ==========================================
st.title("🛒 Trợ lý AI Phân tích Hành vi Khách hàng")
st.markdown("---")

tab1, tab2 = st.tabs(["🎯 Công cụ Dự đoán", "🧠 Giải phẫu Thuật toán LightGBM"])

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
                st.warning("Không có mô hình AI để chạy. Vui lòng khắc phục lỗi nạp file ở trên.")
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
# TAB 2: MINH HỌA THUẬT TOÁN (SỬ DỤNG PLOTLY)
# ---------------------------------------------------------
with tab2:
    st.subheader("🕵️ Giải phẫu \"Bộ não\" của LightGBM")
    st.markdown("""
    Làm sao một con AI có thể biết trước khách hàng sẽ mua hay thoát? Dưới đây là bức tranh toàn cảnh 
    về những yếu tố cốt lõi định hình quyết định của thuật toán.
    """)
    
    if model is not None:
        try:
            # 1. Trích xuất Feature Importance và SẮP XẾP CHUẨN
            importance_values = model.feature_importances_
            feature_names = model.feature_name_
            
            df_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_values
            })
            
            # Chỉ lấy Top 10 yếu tố mạnh nhất và sắp xếp giảm dần để vẽ đẹp hơn
            df_top10 = df_importance.sort_values(by='Importance', ascending=True).tail(10)
            
            # 2. Vẽ biểu đồ tương tác bằng Plotly
            st.markdown("#### 🏆 Top 10 Cột trụ chốt sale (Tính năng Quan trọng nhất)")
            fig = px.bar(
                df_top10, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                text='Importance',
                color='Importance',
                color_continuous_scale='Blues',
                labels={'Importance': 'Mức độ Tác động (Điểm)', 'Feature': 'Hành vi Khách hàng'}
            )
            
            # Tối ưu giao diện biểu đồ
            fig.update_traces(textposition='outside')
            fig.update_layout(
                xaxis_title="Mức độ Tác động",
                yaxis_title="",
                plot_bgcolor='rgba(0,0,0,0)', # Nền trong suốt
                height=500
            )
            
            # Hiển thị biểu đồ lên Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. Phân tích Insight Kinh doanh Tự động
            st.markdown("#### 💡 Đọc vị Insight Kinh doanh từ Biểu đồ")
            
            top_1_feature = df_top10.iloc[-1]['Feature']
            top_2_feature = df_top10.iloc[-2]['Feature']
            
            st.info(f"""
            **1. Yếu tố chí mạng:** Thuật toán phát hiện ra rằng **{top_1_feature}** là yếu tố số 1 quyết định sinh tử của một đơn hàng. 
            Mọi chiến dịch tối ưu hóa Website (UX/UI) cần phải dồn tài nguyên để cải thiện chỉ số này đầu tiên.
            """)
            
            st.success(f"""
            **2. Yếu tố giữ chân:** Đứng ở vị trí thứ hai là **{top_2_feature}**. Việc theo dõi chặt chẽ hành vi này 
            sẽ giúp đội ngũ Marketing xác định chính xác thời điểm nào nên bung mã khuyến mãi để khách không thoát trang.
            """)
            
            st.warning("""
            **3. Nhóm yếu tố vô hình:** Ngược lại, các yếu tố như *Trình duyệt (Browser)* hay *Hệ điều hành (OperatingSystems)* 
            lại có điểm số rất thấp. Điều này cho thấy khách hàng mua sắm vì nhu cầu và trải nghiệm trên trang, chứ không bị rào cản bởi công nghệ họ đang xài.
            """)
            
        except Exception as e:
            st.error(f"Không thể vẽ biểu đồ Feature Importance: {e}")
    else:
        st.warning("Vui lòng tải mô hình thành công để xem phân tích.")
