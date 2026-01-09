import numpy as np

def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    X_train = np.zeros((n_train, max_train_card))
    y_train = np.zeros((n_train, 1))

    for i in range(n_train):
        # 1. Tirer une longueur au hasard entre 1 et 10
        cardinality = np.random.randint(1, max_train_card + 1)
        
        # 2. Créer les données (chiffres entre 1 et 10)
        data = np.random.randint(1, 11, size=cardinality)
        
        # 3. Remplir avec Left Padding
        X_train[i, -cardinality:] = data
        
        # 4. Calculer la somme
        y_train[i] = np.sum(data)

    return X_train, y_train


def create_test_dataset():
    # Consigne : return a list of numpy arrays
    X_test_list = []
    y_test_list = []
    
    # Les cardinalités vont de 5 à 100 par pas de 5
    cardinalities = range(5, 105, 5) 
    n_samples_per_card = 10000

    for card in cardinalities:
        X_batch = np.random.randint(1, 11, size=(n_samples_per_card, card))
        
        # Calculer la somme
        y_batch = np.sum(X_batch, axis=1, keepdims=True)
        
        # Ajouter à la liste
        X_test_list.append(X_batch)
        y_test_list.append(y_batch)

    return X_test_list, y_test_list