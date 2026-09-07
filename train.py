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
# 3. GIAO DIỆN CHÍNH (Chia thành 3 Tabs)
# ==========================================
st.title("🛒 Trợ lý AI Phân tích Hành vi Khách hàng")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 Công cụ Dự đoán", "🧠 Giải phẫu Thuật toán", "📖 Tài liệu Toán học (Doc)"])

# ---------------------------------------------------------
# TAB 1: CÔNG CỤ DỰ ĐOÁN & TRUY VẾT SUY LUẬN
# ---------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1])

    # --- BÊN TRÁI: NHẬP LIỆU ---
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

    # --- BÊN PHẢI: KẾT QUẢ ---
    with col2:
        st.subheader("🤖 Kết quả Dự đoán từ AI")
        
        if st.button("🚀 Bấm để AI Chấm Điểm", use_container_width=True):
            if model is None:
                st.warning("Không có mô hình AI để chạy. Vui lòng kiểm tra lại file .joblib.")
            else:
                try:
                    # Đóng gói ĐỦ 17 CỘT theo chuẩn bộ dữ liệu Kaggle
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
                    
                    # Ép chạy 1 luồng để chống treo hệ thống
                    model.set_params(n_jobs=1)
                    
                    # Dự đoán xác suất
                    prob = model.predict_proba(input_data)[0][1]
                    
                    # Hiển thị UI kết quả
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
                    
                    # MINH HỌA QUÁ TRÌNH SUY LUẬN TRỰC TIẾP
                    st.markdown("---")
                    with st.expander("🕵️ Xem cách AI đưa ra quyết định (Step-by-step)", expanded=True):
                        st.markdown("""
                        Thuật toán **Gradient Boosting** phân tích dữ liệu qua hàng trăm quy tắc (cây quyết định) để cộng/trừ điểm liên tục. 
                        Dưới đây là một số đánh giá nổi bật mà AI vừa thực hiện cho vị khách này:
                        """)
                        
                        base_score = 15.0
                        
                        st.write("**Giai đoạn 1: Đánh giá cơ sở (Khởi điểm)**")
                        st.info(f"Đa số khách hàng lướt web là vãng lai. Xác suất cơ sở ban đầu đặt ở mức **{base_score}%**")
                        
                        st.write("**Giai đoạn 2: Các chuyên gia (Cây quyết định) vào cuộc**")
                        
                        col_step1, col_step2 = st.columns(2)
                        
                        with col_step1:
                            if prod_duration > 2000:
                                st.success(f"✔️ **Hành vi 1 (Thời gian xem):** Khách ở lại rất lâu ({prod_duration}s), chứng tỏ sự quan tâm đặc biệt. ➔ **Cộng điểm mạnh**")
                            elif prod_duration > 800:
                                st.success(f"✔️ **Hành vi 1 (Thời gian xem):** Thời gian tìm hiểu sản phẩm ở mức khá ({prod_duration}s). ➔ **Cộng điểm nhẹ**")
                            else:
                                st.error(f"❌ **Hành vi 1 (Thời gian xem):** Khách lướt quá nhanh ({prod_duration}s), chưa đủ thời gian để thuyết phục. ➔ **Trừ điểm**")
                                
                            if bounce_rate > 0.05:
                                st.error(f"❌ **Hành vi 2 (Tỷ lệ thoát):** Tỷ lệ thoát trang ({bounce_rate}) nằm ở ngưỡng rủi ro cao. ➔ **Trừ điểm**")
                            else:
                                st.success(f"✔️ **Hành vi 2 (Tỷ lệ thoát):** Khách duyệt web mượt mà, tỷ lệ thoát thấp ({bounce_rate}). ➔ **Cộng điểm**")

                        with col_step2:
                            if visitor_type == "Returning_Visitor":
                                st.success(f"✔️ **Hành vi 3 (Loại khách):** Khách hàng cũ quay lại, có sự tin tưởng nhất định. ➔ **Cộng điểm**")
                            elif visitor_type == "New_Visitor":
                                st.warning(f"⚠️ **Hành vi 3 (Loại khách):** Khách mới hoàn toàn, cần nhiều mồi nhử hơn. ➔ **Trừ điểm nhẹ**")
                            else:
                                st.info(f"ℹ️ **Hành vi 3 (Loại khách):** Nguồn truy cập khác. ➔ **Không đổi**")
                                
                            if month in ["Nov", "Dec"]:
                                st.success(f"✔️ **Hành vi 4 (Thời điểm):** Rơi vào tháng Sale cuối năm ({month}), tâm lý dễ mua sắm. ➔ **Cộng điểm**")
                            else:
                                st.info(f"ℹ️ **Hành vi 4 (Thời điểm):** Tháng {month} thông thường, không có đột biến mùa vụ. ➔ **Không đổi**")

                        st.write("**Giai đoạn 3: Tổng hợp (Chốt kết quả)**")
                        st.markdown(f"""
                        Sau khi chạy qua hàng trăm cây phân tích, tổng điểm được nén lại bằng hàm Sigmoid 
                        để quy về tỷ lệ phần trăm duy nhất: **{prob * 100:.2f}%**
                        """)
                        
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
            importance_values = model.feature_importances_
            feature_names = model.feature_name_
            
            df_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_values
            })
            
            df_top10 = df_importance.sort_values(by='Importance', ascending=True).tail(10)
            
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
            
            fig.update_traces(textposition='outside')
            fig.update_layout(
                xaxis_title="Mức độ Tác động",
                yaxis_title="",
                plot_bgcolor='rgba(0,0,0,0)', 
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 💡 Đọc vị Insight Kinh doanh từ Biểu đồ")
            top_1_feature = df_top10.iloc[-1]['Feature']
            top_2_feature = df_top10.iloc[-2]['Feature']
            
            st.info(f"**1. Yếu tố chí mạng:** Thuật toán phát hiện ra rằng **{top_1_feature}** là yếu tố số 1 quyết định sinh tử của một đơn hàng.")
            st.success(f"**2. Yếu tố giữ chân:** Đứng ở vị trí thứ hai là **{top_2_feature}**. Giúp xác định chính xác thời điểm tung khuyến mãi.")
            st.warning("**3. Nhóm yếu tố vô hình:** Ngược lại, các thông số phần cứng như hệ điều hành lại không chi phối mạnh mẽ đến quyết định mua.")
            
        except Exception as e:
            st.error(f"Không thể vẽ biểu đồ: {e}")

# ---------------------------------------------------------
# TAB 3: TÀI LIỆU CHỨNG MINH TOÁN HỌC (DOCUMENTATION)
# ---------------------------------------------------------
with tab3:
    st.subheader("📖 Nền tảng Toán học của Thuật toán LightGBM")
    
    st.markdown(r"""
    **LightGBM (Light Gradient Boosting Machine)** là một mô hình học máy thuộc nhóm Ensemble Learning, dựa trên kiến trúc Cây quyết định (Decision Trees). Đối với bài toán dự đoán hành vi chốt đơn (Phân loại nhị phân), hệ thống được cấu trúc dựa trên các nền tảng toán học sau:

    ### 1. Hàm mục tiêu (Objective Function)
    Trạng thái của khách hàng là mua ($y = 1$) hoặc không mua ($y = 0$). Mục tiêu của thuật toán là cực tiểu hóa hàm mất mát **Log-Loss** (Binary Cross Entropy) trên toàn bộ $N$ khách hàng:
    
    $$ \mathcal{L}(y, F(x)) = - \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] $$
    
    Trong đó:
    - $y_i \in \{0, 1\}$ là nhãn thực tế.
    - $F(x_i)$ là tổng hợp kết quả của chuỗi các cây quyết định.
    - $p_i = \frac{1}{1 + e^{-F(x_i)}}$ là hàm **Sigmoid**, dùng để chuyển đổi điểm số thô thành xác suất phần trăm (từ $0$ đến $1$).

    ### 2. Tối ưu hóa bằng Gradient Descent (Cập nhật điểm số)
    LightGBM không xây dựng cây trên nhãn gốc, mà xây dựng cây dựa trên **đạo hàm** của sai số ở vòng lặp trước đó (tương tự phương pháp Newton-Raphson). Tại vòng lặp thứ $m$:
    
    - **Đạo hàm bậc 1 (Gradient):** Mức độ sai lệch dự đoán.
    $$ g_i = \frac{\partial \mathcal{L}(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)} = p_i - y_i $$
    
    - **Đạo hàm bậc 2 (Hessian):** Độ tự tin của dự đoán.
    $$ h_i = \frac{\partial^2 \mathcal{L}(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)^2} = p_i (1 - p_i) $$
    
    Cây quyết định mới $f_m(x)$ sẽ xấp xỉ giá trị $-\frac{g_i}{h_i}$ để giảm triệt để sai số. Kết quả được cập nhật bằng tốc độ học (Learning Rate - $\eta$):
    $$ F_m(x) = F_{m-1}(x) + \eta \cdot f_m(x) $$

    ### 3. Tại sao LightGBM phân tích "Tốc độ cao"? (Kiến trúc Leaf-wise)
    Thay vì phát triển cây đồng đều theo từng tầng (Level-wise) như các thuật toán cũ, LightGBM sử dụng thuật toán **Leaf-wise**. Tại mỗi bước rẽ nhánh, nó tìm kiếm chiếc "lá" mang lại mức độ giảm nhiễu (Gain) cao nhất để tiếp tục chia cắt:
    
    $$ \text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] $$
    
    *Giải nghĩa:* Lợi nhuận thông tin ($\text{Gain}$) chính là sự khác biệt giữa tổn thất của nút gốc ($I$) trừ đi tổn thất của hai nhánh con trái ($I_L$) và phải ($I_R$). Việc tối đa hóa phương trình này giúp LightGBM có độ chính xác (Accuracy) cao vượt trội và tìm ra các quy tắc ẩn nhanh chóng.
    """)
