import pandas as pd
from scipy.stats import skew


class LRFMSPreprocessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.lrfms = None
        self.lrfms_scaled = None
        self.X = None

    # ======== Đọc dữ liệu ========
    def read_data(self):
        self.df = pd.read_excel(self.file_path)
        return self.df

    # ======== Làm sạch ========
    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    @staticmethod
    def filter_invalid(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].dropna()

    @staticmethod
    def add_total_amount(df: pd.DataFrame) -> pd.DataFrame:
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
        return df

    @staticmethod
    def handle_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
        q_low, q_high = df[column].quantile([0.1, 0.9])
        df[column] = df[column].clip(q_low, q_high)
        return df

    # ======== Chuẩn hóa thủ công ========
    @staticmethod
    def manual_standardize(df: pd.DataFrame, cols):
        df_scaled = df.copy()
        for col in cols:
            mean = df[col].mean()
            std = df[col].std(ddof=0)
            df_scaled[col] = (df[col] - mean) / std
        return df_scaled

    # ======== Tính LRFMS ========
    @staticmethod
    def calculate_lrfms(df: pd.DataFrame) -> pd.DataFrame:
        """
        Giả định:
        - Loyalty = tổng số lần mua / tổng tháng hoạt động
        - Recency = số ngày kể từ lần mua gần nhất
        - Frequency = số đơn hàng
        - Monetary = tổng tiền chi tiêu
        - Satisfaction = trung bình TotalAmount mỗi hóa đơn
        """

        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        agg = df.groupby("CustomerID").agg({
            "InvoiceDate": ["min", "max", "nunique"],
            "InvoiceNo": "nunique",
            "TotalAmount": ["sum", "mean"]
        })

        agg.columns = [
            "FirstPurchase", "LastPurchase", "ActiveDays",
            "Frequency", "Monetary", "Satisfaction"
        ]
        agg = agg.reset_index()

        # Loyalty = tần suất giao dịch trung bình theo thời gian hoạt động
        agg["Loyalty"] = agg["Frequency"] / (
            ((agg["LastPurchase"] - agg["FirstPurchase"]).dt.days / 30).replace(0, 1)
        )

        # Recency
        agg["Recency"] = (reference_date - agg["LastPurchase"]).dt.days

        # Giữ lại đúng cột cần thiết
        lrfms = agg[["CustomerID", "Loyalty", "Recency", "Frequency", "Monetary", "Satisfaction"]]
        return lrfms

    # ======== Xử lý tổng thể ========
    def process(self):
        df = self.read_data()
        df = self.remove_duplicates(df)
        df = self.filter_invalid(df)
        df = self.add_total_amount(df)
        df = self.handle_outliers(df, "TotalAmount")

        skew_value = skew(df["TotalAmount"])
        print(f"Skewness of TotalAmount: {skew_value:.2f}")

        self.lrfms = self.calculate_lrfms(df)
        self.lrfms_scaled = self.manual_standardize(
            self.lrfms, ["Loyalty", "Recency", "Frequency", "Monetary", "Satisfaction"]
        )
        self.X = self.lrfms_scaled[["Loyalty", "Recency", "Frequency", "Monetary", "Satisfaction"]].to_numpy()

        return self.X,self.lrfms,self.lrfms_scaled


