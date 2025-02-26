# import pickle

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np


# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()

# -------------------------------------------------------------

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Get maximum length of sequences
max_len = max(len(seq) for seq in data_dict['data'])

# Pad sequences to make them uniform length
processed_data = []
for seq in data_dict['data']:
    if isinstance(seq, np.ndarray):
        # If sequence is already numpy array, flatten it
        seq_flat = seq.flatten()
    else:
        # If sequence is list or other type, convert to numpy and flatten
        seq_flat = np.array(seq).flatten()
    
    # Pad or truncate to max_len
    if len(seq_flat) > max_len:
        processed_data.append(seq_flat[:max_len])
    else:
        # Pad with zeros
        padded = np.zeros(max_len)
        padded[:len(seq_flat)] = seq_flat
        processed_data.append(padded)

# Convert to numpy array
data = np.array(processed_data)
labels = np.array(data_dict['labels'])

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, 
    labels, 
    test_size=0.2, 
    shuffle=True, 
    stratify=labels
)

# Initialize and train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)