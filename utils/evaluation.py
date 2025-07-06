from sklearn.metrics import classification_report

def metric_report(y_true ,y_pred):
    return print(classification_report(y_true ,y_pred))
