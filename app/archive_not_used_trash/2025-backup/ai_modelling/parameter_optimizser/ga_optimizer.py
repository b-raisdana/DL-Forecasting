from deap import algorithms
import random
import time
import numpy as np
import tensorflow as tf
from deap import base, creator, tools

from ai_modelling.cnn_lstm_attention.cnn_lstm_attention_model import build_cnn_lstm_attention_model
from ai_modelling.dataset_generator.ram_batch import build_ram_dataset

# Assuming you have your build_cnn_lstm_attention_model defined
# and train_model function to train model on data and return val loss.

# Define hyperparameter ranges
PARAM_RANGES = {
    'cnn_filters': (32, 128),
    'lstm_units_1': (16, 128),
    'lstm_units_2': (16, 128),
    'cnn_count': (1, 4),
    'kernel_step': (1, 4),
    'dropout_rate': (0.1, 0.5),
    'num_heads': (2, 8),
    'key_dim': (8, 32),
}

# Example input_shapes dictionary to pass to model builder
input_shapes = {
    'structure': (96, 5),
    'pattern': (168, 5),
    'trigger': (672, 5),
    'double': (100, 5)
}

y_len = 6

_train_dataset = None
def dataset():
    global _train_dataset
    if _train_dataset is None:
        _train_dataset = build_ram_dataset(batch_size=80)
    return _train_dataset

# Define a simple training function (you should adapt with your own data)
def train_model( params, time_limit=2* 3600):
    """
    Build and train the model with given params.
    Stops training if time_limit seconds exceeded.
    Returns validation loss (fitness).
    """

    cnn_filters = params[0]
    lstm_units = [params[1], params[2]]
    cnn_count = params[3]
    kernel_step = params[4]
    dropout_rate = params[5]
    num_heads = params[6]
    key_dim = params[7]

    model = build_cnn_lstm_attention_model(
        y_len=y_len,
        input_shapes=input_shapes,
        cnn_filters=cnn_filters,
        lstm_units=lstm_units,
        cnn_count=cnn_count,
        kernel_step=kernel_step,
        dropout_rate=dropout_rate,
        num_heads=num_heads,
        key_dim=key_dim,
    )

    model.compile(optimizer='adam', loss='mse')

    start_time = time.time()
    class TimeLimitCallback(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if time.time() - start_time > time_limit:
                self.model.stop_training = True

    model.fit(dataset().take(1), steps_per_epoch=1, epochs=1, verbose=0)
    history = model.fit(
        dataset(),
        validation_data=dataset(),
        epochs=1000,
        batch_size=200,
        steps_per_epoch=100,
        validation_steps=20,
        callbacks=[TimeLimitCallback(),])

    best_val_loss = min(history.history['val_loss'])

    return best_val_loss

def ga_setup():
    # GA setup using DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # minimize val_loss
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Attribute generators
    toolbox.register("cnn_filters", random.randint, *PARAM_RANGES['cnn_filters'])
    toolbox.register("lstm_units_1", random.randint, *PARAM_RANGES['lstm_units_1'])
    toolbox.register("lstm_units_2", random.randint, *PARAM_RANGES['lstm_units_2'])
    toolbox.register("cnn_count", random.randint, *PARAM_RANGES['cnn_count'])
    toolbox.register("kernel_step", random.randint, *PARAM_RANGES['kernel_step'])
    toolbox.register("dropout_rate", random.uniform, *PARAM_RANGES['dropout_rate'])
    toolbox.register("num_heads", random.randint, *PARAM_RANGES['num_heads'])
    toolbox.register("key_dim", random.randint, *PARAM_RANGES['key_dim'])

    # Structure initializers
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.cnn_filters,
                      toolbox.lstm_units_1,
                      toolbox.lstm_units_2,
                      toolbox.cnn_count,
                      toolbox.kernel_step,
                      toolbox.dropout_rate,
                      toolbox.num_heads,
                      toolbox.key_dim), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)

    # Genetic operators
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.decorate("mutate", check_bounds(PARAM_RANGES))

    return toolbox
# Evaluation function
def evaluate(individual):
    val_loss = train_model(individual, time_limit=3600)
    print(f"Evaluated individual {individual} with val_loss={val_loss:.4f}")
    return (val_loss,)


# Mutation constraints
def check_bounds(min_max):
    def decorator(func):
        def wrapper(individual, *args, **kwargs):
            func(individual, *args, **kwargs)
            # Clamp values within bounds
            for i, key in enumerate(PARAM_RANGES.keys()):
                mn, mx = PARAM_RANGES[key]
                if individual[i] < mn:
                    individual[i] = mn
                elif individual[i] > mx:
                    individual[i] = mx
                # Round integer params
                if isinstance(mn, int):
                    individual[i] = int(round(individual[i]))
                else:
                    individual[i] = float(individual[i])
            return individual
        return wrapper
    return decorator


# Run GA optimization
def run_ga(toolbox, pop_size=6, generations=5):
    pop = toolbox.population(n=pop_size)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    pop, log = algorithms.eaSimple(pop, toolbox,
                                  cxpb=0.5, mutpb=0.3,
                                  ngen=generations,
                                  stats=stats, halloffame=hof,
                                  verbose=True)
    print("Best individual is ", hof[0], "with fitness", hof[0].fitness.values)
    return hof[0]



if __name__ == "__main__":
    toolbox = ga_setup()
    run_ga(toolbox)