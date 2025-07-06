from utils.data_preprocessing import(load_data,preprocess_data,split_data)
from utils.evaluation import(metric_report)
from models.traditional_ml import (train_kneighbors,train_logistic,train_decisiontree,train_randomforest,make_predictions)
from models.neural_networks import (neural_model,train_neural_network)
import matplotlib.pyplot as plt



def main():
    print("Loading data...............................")
    data = load_data('loan_data1.csv')
    processed_data = preprocess_data(data)

    X = processed_data.drop('loan_status',axis=1)
    y = processed_data['loan_status']

    X_train, X_test, y_train, y_test = split_data(X, y)


    print ("\nTraining KNeighbors Classifier ...")
    kn_model = train_kneighbors(X_train,y_train)
    kn_pred = make_predictions(kn_model,X_test)
    print ("\nKNeighborsClassifier Results :")
    metric_report(y_test,kn_pred)


    print ("\nTraining Logistic Regression ...")
    lg_model = train_logistic(X_train,y_train)
    lg_pred = make_predictions(lg_model,X_test)
    print ("\nLogistic Regression Results :")
    metric_report(y_test,lg_pred)


    print ("\nTraining DecisionTree Classifier ...")
    dtc_model = train_decisiontree(X_train,y_train)
    dtc_pred = make_predictions(dtc_model,X_test)
    print ("\nDecisionTree Classifier Results :")
    metric_report(y_test,dtc_pred)


    print ("\nTraining Random Forest Classifier ...")
    rf_model = train_randomforest(X_train,y_train)
    rf_pred = make_predictions(rf_model,X_test)
    print ("\nRandom Forest Classifier Results :")
    metric_report(y_test,rf_pred)


    
    print ("\nTraining Neural Network ... ")
    nn_model = neural_model(input_dimension= 13)
    history = train_neural_network(nn_model, X_train, y_train)
    nn_pred =nn_model.predict(X_test)
    nn_pred =( nn_pred > 0.5).astype(int)
    print("\nNeural Network Results :")
    metric_report(y_test ,nn_pred)



    # Plot loss vs val_loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs Val_Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.ylim(0.0,1.0)
    plt.legend()
    plt.show() 



if __name__=="__main__":
    main()







