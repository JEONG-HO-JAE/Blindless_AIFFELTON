from tensorflow.keras import backend as K


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def accuracy(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))

    return (true_positives + true_negatives) / (possible_positives + possible_negatives + K.epsilon())

def print_evaluation(model, generator):
    # Evaluate the model on the test data
    evaluation = model.evaluate(generator)

    # Extract individual metric values from the evaluation result
    loss_value = evaluation[0]
    sensitivity_value = evaluation[1]
    specificity_value = evaluation[2]
    accuracy_value = evaluation[3]

    # Print the metric values
    print(f"Loss: {loss_value}")
    print(f"Sensitivity: {sensitivity_value}")
    print(f"Specificity: {specificity_value}")
    print(f"Accuracy: {accuracy_value}")