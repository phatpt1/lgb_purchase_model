import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score

def main():
    csv_file = "online_shoppers.csv"
    
    # 1. KIỂM TRA & NẠP DỮ LIỆU
    if not os.path.exists(csv_file):
        print(f"Lỗi: Không tìm thấy file '{csv_file}'. Vui lòng để chung thư mục với script.")
        return
        
    print(f"1. Đang nạp dữ liệu từ {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"   -> Tổng số dòng: {len(df)}")
    
    # Bộ Online Shoppers có cột nhãn là 'Revenue' (True/False)
    # Ta chuyển nhãn về dạng số nhị phân: 1 (Mua) và 0 (Không mua)
    df["Revenue"] = df["Revenue"].astype(int)
    
    # 2. KHAI BÁO CỘT CATEGORICAL (Siêu quan trọng cho LightGBM)
    # Đây là các cột chứa chữ (String/Object) hoặc dữ liệu phân loại trong bộ Kaggle
    cat_features = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
    
    print("2. Định dạng kiểu dữ liệu Categorical...")
    for col in cat_features:
        # Nếu bộ dữ liệu Kaggle có chứa NaN trong các cột này, chuyển chúng thành dạng string 'Missing'
        # để tránh Pandas gặp lỗi khi ép kiểu category
        df[col] = df[col].fillna("Missing") 
        df[col] = df[col].astype("category")
    
    # 3. CHIA TÁCH DỮ LIỆU (TRAIN / TEST)
    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]
    
    print("3. Phân tách tập Train (80%) và Test (20%)...")
    # Sử dụng stratify=y để đảm bảo tỷ lệ 15% khách mua hàng được rải đều ở cả 2 tập
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. TÍNH TOÁN TRỌNG SỐ CHO LỚP THIỂU SỐ (Khách thực sự Mua hàng)
    # Công thức: (Số lượng Khách KHÔNG mua) / (Số lượng Khách MUA)
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"   -> Đã gán trọng số tự động (scale_pos_weight = {pos_weight:.2f})")

    # 5. KHỞI TẠO VÀ HUẤN LUYỆN LIGHTGBM
    # Cấu hình "Tiny Model": Giới hạn max_depth và num_leaves để file model cực nhỏ
    model = lgb.LGBMClassifier(
        n_estimators=120,         # Số lượng cây rẽ nhánh
        max_depth=5,              # Độ sâu tối đa của mỗi cây (ngăn Overfitting)
        num_leaves=25,            # Số lượng lá tối đa
        learning_rate=0.05,       # Tốc độ học (nhỏ thì hội tụ chậm nhưng chắc chắn)
        scale_pos_weight=pos_weight, # Ép mô hình tập trung vào nhóm mua hàng
        random_state=42,
        n_jobs=-1                 # Dùng toàn bộ core CPU của máy local
    )
    
    print("4. Bắt đầu huấn luyện mô hình LightGBM...")
    model.fit(
        X_train, y_train,
        categorical_feature=cat_features # Báo cho mô hình biết đâu là cột danh mục
    )
    
    # 6. ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST
    print("\n" + "="*40)
    print("BÁO CÁO KẾT QUẢ TRÊN TẬP TEST (20% DỮ LIỆU MỚI)")
    print("="*40)
    
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # PR-AUC là thước đo phản ánh thực tế tốt nhất cho bài toán imbalanced
    print(f"PR-AUC Score (Area Under Precision-Recall Curve): {average_precision_score(y_test, y_pred_prob):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}\n")
    print("Ma trận phân loại:")
    print(classification_report(y_test, y_pred, target_names=["Không Mua (0)", "Mua Hàng (1)"]))
    
    # Xem 5 yếu tố quan trọng nhất quyết định việc mua hàng
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nTop 5 yếu tố quan trọng nhất:")
    print(importance_df.head(5).to_string(index=False))
    
    # 7. XUẤT FILE MODEL
    model_filename = "lgb_purchase_model.joblib"
    joblib.dump(model, model_filename, compress=3)
    
    file_size_kb = os.path.getsize(model_filename) / 1024
    print("\n" + "="*40)
    print(f"HOÀN TẤT! Mô hình đã được lưu tại: {model_filename}")
    print(f"Dung lượng file: {file_size_kb:.1f} KB (Sẵn sàng cho API Deployment)")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore") # Tắt các cảnh báo lặt vặt của pandas/scikit-learn
    main()