import json
import os


def pre_save_model_specs(
    fp: str = None,
    model_name: str = None,
    image_size: str = None,
    inital_json_grab: int = None,
    unique_classes: int = None,
    learning_rate: float = None,
    beta_1: float = None,
    beta_2: float = None,
    loss: str = None,
    metrics: list[str] = None,
    img_width: int = None,  # this is what the model expects for the input layer
    img_height: int = None,  # this is what the model expects for the input layer
) -> None:
    specs = {
        "model_name": model_name,
        "image_size": image_size,
        "initial_json_grab": inital_json_grab,
        "unique_classes": unique_classes,
        "learning_rate": learning_rate,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "loss": loss,
        "metrics": metrics,
        "preprocessed_image_dimensions": f"{img_width}x{img_height}",
    }

    with open(fp["SPECS"], "w") as f:
        json.dump(specs, f, indent=4)


def post_save_model_specs(
    fp: str = None,
    training_time: str = None,  # I am probably wrong about this datatype but whatever
    loss: float = None,
    accuracy: float = None,
    model=None,  # the datatype isn’t important, but it is probably helpful
) -> None:
    # Read the existing specs from the JSON file
    with open(fp["SPECS"], "r") as f:
        specs = json.load(f)

    # Update the specs with new attributes
    specs.update(
        {
            "training_time": training_time,
            "final_loss": loss,
            "final_accuracy": accuracy,
            "model_summary": str(model.summary()) if model else None,
        }
    )

    # Write the updated specs back to the JSON file
    with open(fp["SPECS"], "w") as f:
        json.dump(specs, f, indent=4)
