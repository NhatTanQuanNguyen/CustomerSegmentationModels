import pandas as pd
from scipy.stats import skew


class RFMPreprocessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.rfm = None
        self.rfm_scaled = None
        self.X = None

    def read_data(self):
        self.df = pd.read_excel(self.file_path)
        return self.df

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    @staticmethod
    def filter_invalid(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].dropna()

    @staticmethod
    def add_total_amount(df: pd.DataFrame) -> pd.DataFrame:
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
        return df.drop(columns=["Quantity", "UnitPrice"])

    @staticmethod
    def handle_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
        q_low, q_high = df[column].quantile([0.1, 0.9])
        df[column] = df[column].clip(q_low, q_high)
        return df

    @staticmethod
    def manual_standardize(df: pd.DataFrame, cols):
        df_scaled = df.copy()
        for col in cols:
            mean = df[col].mean()
            std = df[col].std(ddof=0)  # giống sklearn
            df_scaled[col] = (df[col] - mean) / std
        return df_scaled

    @staticmethod
    def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        rfm = df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (reference_date - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalAmount": "sum"
        }).reset_index()
        rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
        return rfm

    def process(self):
        df = self.read_data()
        df = self.remove_duplicates(df)
        df = self.filter_invalid(df)
        df = self.add_total_amount(df)
        df = self.handle_outliers(df, "TotalAmount")

        skew_value = skew(df["TotalAmount"])
        print(f"Skewness of TotalAmount: {skew_value:.2f}")

        self.rfm = self.calculate_rfm(df)
        self.rfm_scaled = self.manual_standardize(
            self.rfm, ["Recency", "Frequency", "Monetary"]
        )
        self.X = self.rfm_scaled[["Recency", "Frequency", "Monetary"]].to_numpy()

        return self.X
