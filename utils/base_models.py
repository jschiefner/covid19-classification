import tensorflow
# models


func_names = [m.__name__ for m in tensorflow.keras.applications]
funcs_dict = dict(zip(func_names, tensorflow.keras.applications))

def get_all_model_names():
    return func_names

def get_model_by_name(m):
    if not m in funcs_dict:
        return None
    return funcs_dict[m]


