
import data_providers as gen
import model_storage as storage
import numpy as np
import data_visualizer

def evaluate(model_name):
    """
    Evaluates the model stored in the specified file.

    Parameters
    ----------

    model_name : string
        The name of the file to read the model from
    """

    model = storage.load_model(model_name)

    model.summary()

    score = model.evaluate_generator(gen.validation_data("data\\validation"), steps=2)

    print (model.metrics_names)
    print (score)


    predictions = model.predict_generator(gen.validation_data("data\\validation"), steps=10)
    data_visualizer.draw_bounding_boxes("data\\validation", predictions, "data\\results")