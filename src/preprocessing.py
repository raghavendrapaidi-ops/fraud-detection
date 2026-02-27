from sklearn.preprocessing import RobustScaler


def scale_data(df):
    scaler = RobustScaler()

    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])

    return df
