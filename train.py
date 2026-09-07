import streamlit as st
import pandas as pd
import joblib

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
# 3. GIAO DIỆN CHÍNH
# ==========================================
st.title("🛒 Trợ lý AI Phân tích Hành vi Khách hàng")
st.markdown("---")

col1, col2 = st.columns([1, 1])

# --- PHẦN NHẬP LIỆU BÊN TRÁI ---
with col1:
    st.subheader("📝 Nhập dữ liệu hành vi của khách")
    
    # Các thanh trượt mô phỏng hành vi lướt web
    prod_duration = st.slider("⏱️ Thời gian xem Sản phẩm (giây)", 0, 5000, 1500)
    admin_duration = st.slider("⚙️ Thời gian ở trang Quản lý/Thanh toán", 0, 1000, 50)
    bounce_rate = st.slider("🏃 Tỷ lệ thoát nhanh (Bounce Rate)", 0.0, 0.2, 0.01)
    
    # Các lựa chọn dạng danh mục
    col_a, col_b = st.columns(2)
    with col_a:
        month = st.selectbox("Tháng truy cập", ["Feb", "Mar", "May", "Oct", "Nov", "Dec"], index=4)
        visitor_type = st.selectbox("Loại khách hàng", ["New_Visitor", "Returning_Visitor", "Other"])
    with col_b:
        weekend = st.radio("Truy cập vào cuối tuần?", [True, False])
        region = st.selectbox("Khu vực địa lý (Region ID)", [1, 2, 3, 4, 5, 6, 7, 8, 9])

# --- PHẦN XỬ LÝ VÀ HIỂN THỊ BÊN PHẢI ---
with col2:
    st.subheader("🤖 Kết quả Dự đoán từ AI")
    
    # Chỉ chạy logic khi người dùng bấm nút
    if st.button("🚀 Bấm để AI Chấm Điểm", use_container_width=True):
        
        # Nếu model không load được từ bước 2, dừng ngay lập tức
        if model is None:
            st.warning("Không có mô hình AI để chạy. Vui lòng khắc phục lỗi nạp file ở trên.")
        else:
            try:
                # 1. Đóng gói dữ liệu đầu vào thành DataFrame
                # Gán sẵn các cột không đưa lên giao diện bằng giá trị mặc định lúc train
                input_data = pd.DataFrame([{
                    "Administrative_Duration": float(admin_duration),
                    "ProductRelated_Duration": float(prod_duration),
                    "BounceRates": float(bounce_rate),
                    "Month": month,
                    "OperatingSystems": 2, 
                    "Browser": 2,
                    "Region": int(region),
                    "TrafficType": 2,
                    "VisitorType": visitor_type,
                    "Weekend": bool(weekend)
                }])
                
                # 2. Xử lý chuẩn xác các cột Categorical (Chống treo hệ thống)
                cat_features = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
                for col in cat_features:
                    # Ép sang dạng chuỗi (string) trước, sau đó mới ép sang category để tránh lỗi ngầm của Pandas
                    input_data[col] = input_data[col].astype(str).astype("category")
                
                # 3. Tính toán xác suất
                # proba trả về array 2 chiều, [0][1] là lấy xác suất của nhãn 1 (Khách Mua)
                prob = model.predict_proba(input_data)[0][1]
                
                # 4. Hiển thị UI kết quả
                st.write("### Xác suất chốt đơn:")
                st.progress(float(prob))
                st.metric(label="", value=f"{prob * 100:.2f}%")
                
                st.markdown("### 💡 Đề xuất hành động:")
                if prob >= 0.7:
                    st.success("**🟢 KHÁCH HÀNG RẤT NÉT!** \n\n Không cần mã giảm giá, cứ để họ tự chốt đơn.")
                elif prob >= 0.4:
                    st.warning("**🟠 KHÁCH ĐANG PHÂN VÂN!** \n\n Nên tung ngay mã giảm giá 10% (Freeship) để chốt sale tức thì.")
                else:
                    st.error("**🔴 KHÁCH VÃNG LAI.** \n\n Xác suất chốt đơn quá thấp. Không nên lãng phí ngân sách hiển thị Ads cho trường hợp này.")
                    
            except Exception as e:
                # Nếu LightGBM gặp lỗi khi đọc dữ liệu, hiển thị hộp thoại đỏ lên màn hình ngay
                st.error(f"❌ Thuật toán gặp lỗi khi dự đoán: {e}")
                st.info("Gợi ý: Hãy kiểm tra lại file train.py xem có thay đổi thứ tự hoặc tên cột nào không.")
