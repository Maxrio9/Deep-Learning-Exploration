import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import randint as sp_randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.metrics import classification_report


# Processing data
dataset = pd.read_csv("cover_data.csv")

features = dataset.iloc[:,:-1]
labels = dataset.iloc[:,-1]

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42, stratify=labels)

#standardize
ct = ColumnTransformer([('standardize', StandardScaler(), ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'])], remainder='passthrough')
scaled_features_train = ct.fit_transform(features_train) #gives numpy arrays
scaled_features_test = ct.transform(features_test) #gives numpy arrays


def design_model():
  # Building model
  model = Sequential(name="Tree_Cover_Model")
  print('Shape of the Input is: {}'.format(scaled_features_train.shape[1]))
  input = tf.keras.Input(shape=(scaled_features_train.shape[1],))
  model.add(input)
  model.add(layers.Dense(64, activation = 'relu'))
  # model.add(layers.Dropout(0.3))
  model.add(layers.Dense(16, activation = 'relu'))
  # model.add(layers.Dropout(0.15))
  model.add(tf.keras.layers.Flatten())
  model.add(layers.Dense(8, activation = 'softmax'))
  opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
  model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], optimizer=opt)
  return model

# print(model.summary())


# Uncomment for Hyperparameter tuning
#-----------------------------------------------------------------------------------------------------------------------
# param_grid = {'batch_size': sp_randint(2,16), 'nb_epoch': sp_randint(10, 100)}
# model = KerasRegressor(build_fn=design_model)
# grid = RandomizedSearchCV(estimator = model, param_distributions = param_grid, scoring = make_scorer(accuracy_score, greater_is_better=False), n_iter = 12)
# grid_result = grid.fit(scaled_features_train, labels_train, verbose = 0)
# print(grid_result)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#-----------------------------------------------------------------------------------------------------------------------



model = design_model()
print(labels_train.shape)
earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=3)
history = model.fit(scaled_features_train, labels_train, callbacks = [earlystop_callback], epochs=12, validation_data=(scaled_features_test, labels_test), verbose=0, batch_size = 32)
print('History:', history.history)

# y_estimate = model.predict(scaled_features_test)
# y_estimate = np.argmax(y_estimate, axis = 1)
# y_true = np.argmax(labels_test, axis = 1)

# print(classification_report(y_true, y_estimate))
