import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():

    df = pd.read_csv(r"C:\Users\jvans\Documents\NOP\Feature_Selection\data\BostonHousing.csv")
    
    # fill missing values
    df = df.fillna(df.mean())

    X = df.drop("medv", axis=1)
    y = df["medv"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test