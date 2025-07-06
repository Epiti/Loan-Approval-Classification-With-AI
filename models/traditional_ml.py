from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def train_kneighbors(X_train,y_train):
    model = KNeighborsClassifier(
        n_neighbors=55
    )
    model.fit(X_train,y_train)
    return model


def train_logistic(X_train,y_train):
    model = LogisticRegression(
        max_iter=1500
    )
    model.fit(X_train,y_train)
    return model


def train_decisiontree(X_train,y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    return model 

def train_randomforest(X_train,y_train):
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    return model

def make_predictions(model, X):
    return model.predict(X)

