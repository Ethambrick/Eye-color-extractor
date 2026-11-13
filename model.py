from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=1000,
        learning_rate_init=0.001
    )

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    print("Model score:", score)

    return model