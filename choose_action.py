import random
import tensorflow as tf


num_servers = 12
output_size = num_servers


models_loaded = False
dqns = []


def load_models(num_vehicles):
    global models_loaded
    for index in range(num_vehicles):
        dqns.append(tf.keras.models.load_model(f'dqn_models/{num_vehicles}_{index}'))
    print('Models loaded!')
    models_loaded = True


def choose_action(num_vehicles, v_index, state):
    if not models_loaded:
        print('Loading models...')
        load_models(num_vehicles)

    dqn = dqns[v_index]
    qualities = dqn(state)
    action = tf.math.argmax(qualities[0])
    action = int(action)

    return action

