
import data_providers as gen
import model_storage as storage
import numpy as np
import data_visualizer

def validate(model_name):
    model = storage.load_model(model_name)

    model.summary()

    score = model.evaluate_generator(gen.validation_data("data\\validation"), steps=2)

    print (model.metrics_names)
    print (score)


    predictions = model.predict_generator(gen.validation_data("data\\validation"), steps=3)
    data_visualizer.draw_bounding_boxes("data\\validation", predictions, "data\\results")