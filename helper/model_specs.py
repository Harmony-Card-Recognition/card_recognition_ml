import os


def pre_save_model_specs(
    specs_filepath: str = None,
    model_name: str = None,
    image_size: str = None,
    inital_json_grab: int = None,
    unique_classes: int = None,
    learning_rate: float = None,
    beta_1: float = None,
    beta_2: float = None,
    loss: str = None,
    metrics: list[str] = None,
    img_width: int = None, # this is what the model expects for the input layer 
    img_height: int = None, # this is what the model expects for the input layer
) -> None:
    with open(specs_filepath, 'w') as f:
        f.write(f'Model Name: {model_name}\n')
        f.write(f'Image Size: {image_size}\n')
        f.write(f'Initial JSON Grab: {inital_json_grab}\n')
        f.write(f'Unique Classes: {unique_classes}\n')
        f.write('\n')
        f.write(f'Learning Rate: {learning_rate}\n')
        f.write(f'Beta 1: {beta_1}\n') 
        f.write(f'Beta 2: {beta_2}\n') 
        f.write(f'Loss: {loss}\n') 
        f.write(f'metrics: {metrics}\n')
        f.write(f'Preprocessed Image Dimensions (wxh): {img_width}x{img_height}')
        f.write('\n')

def post_save_model_specs(
    specs_filepath: str = None,
    training_time: str = None, # I am probably wrong about this datatype but whatever
    loss: float = None,
    accuracy: float = None,
    model = None, # the datatype isn;t important, but it is probably helpful
) -> None:
    with open(specs_filepath, 'a') as f:
        f.write(f'Training Time: {training_time}\n')
        f.write(f'Loss: {loss}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write('\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))
