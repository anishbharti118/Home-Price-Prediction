import json
import pickle
import numpy as np

__model = None
__location = None
__data_columns = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk

    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def get_location_names():
    return __location

def load_saved_artifacts():
    print('Loading saved Artifacts..... start')
    global __data_columns
    global __location
    global __model

    with open("./model/column.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __location = __data_columns[3:]
    
    with open('./model/bangalore_home_prices_model.pickle', 'rb') as f:
        __model = pickle.load(f)
    
    print('Loading saved Artifacts...... done')


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Rajaji Nagar', 2000, 3, 4))
    print(get_estimated_price('Rajaji Nagar', 2000, 4, 3))
    print(get_estimated_price('Noida', 2000, 3, 4))
    print(get_estimated_price('Mumbai', 2000, 3, 4))
