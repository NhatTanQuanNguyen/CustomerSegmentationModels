import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class LRFMSPreprocessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.lrfms = None
        self.lrfms_scaled = None
        self.X = None

    def read_data(self):
        self.df = pd.read_excel(self.file_path)
        return self.df

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(ignore_index=True)

    @staticmethod
    def filter_invalid(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["CustomerID"])
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
        return df

    @staticmethod
    def add_total_amount(df: pd.DataFrame) -> pd.DataFrame:
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
        return df

    @staticmethod
    def handle_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
        q_low, q_high = df[column].quantile([0.05, 0.95])
        df[column] = df[column].clip(q_low, q_high)
        return df

    @staticmethod
    def standardize_with_library(df: pd.DataFrame, cols, method="zscore"):
        scaler = StandardScaler() if method == "zscore" else MinMaxScaler()
        df_scaled = df.copy()
        df_scaled[cols] = scaler.fit_transform(df_scaled[cols])
        return df_scaled

    @staticmethod
    def calculate_lrfms(df: pd.DataFrame) -> pd.DataFrame:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        agg = df.groupby("CustomerID").agg({
            "InvoiceDate": ["min", "max"],
            "InvoiceNo": "nunique",
            "TotalAmount": ["sum", "mean"]
        })
        agg.columns = ["FirstPurchase", "LastPurchase", "Frequency", "Monetary", "Satisfaction"]
        agg = agg.reset_index()
        active_months = ((agg["LastPurchase"] - agg["FirstPurchase"]).dt.days / 30).replace(0, 1)
        agg["Loyalty"] = agg["Frequency"] / active_months
        agg["Recency"] = (reference_date - agg["LastPurchase"]).dt.days
        return agg[["CustomerID", "Loyalty", "Recency", "Frequency", "Monetary", "Satisfaction"]]

    def process(self):
        df = self.read_data()
        df = self.remove_duplicates(df)
        df = self.filter_invalid(df)
        df = self.add_total_amount(df)
        df = self.handle_outliers(df, "TotalAmount")
        skew_value = skew(df["TotalAmount"])
        print(f"Skewness of TotalAmount: {skew_value:.2f}")
        self.lrfms = self.calculate_lrfms(df)
        self.lrfms_scaled = self.standardize_with_library(
            self.lrfms, ["Loyalty", "Recency", "Frequency", "Monetary", "Satisfaction"], method="zscore"
        )
        self.X = self.lrfms_scaled[["Loyalty", "Recency", "Frequency", "Monetary", "Satisfaction"]].to_numpy()
        return self.X, self.lrfms, self.lrfms_scaled
