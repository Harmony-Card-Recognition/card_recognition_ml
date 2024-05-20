from time import localtime, strftime, time
import os
import json

# =======================================
# LOGGING
def log_and_verbose(message, log_dir, verbose):
    # log it in a text file
    with open(os.path.join(log_dir, "log.txt"), "a") as f:
        f.write(f"{get_current_time()}: {message}\n")

    # if verbose, print it
    if verbose: print(message)

# =======================================
# JSON

def get_json_length(json_filepath):
    # Open the JSON file
    with open(json_filepath, 'r', encoding='utf-8') as f:
        # Load the JSON data
        data = json.load(f)

    # Count the number of JSON objects
    num_objects = len(data)

    print(f'There are {num_objects} JSON objects in the file.')
# =======================================
# TIME


def get_current_time():
    """Help: Returns the current time as a nice string."""
    return strftime("%B %d, %I:%M%p", localtime())


def get_elapsed_time(start_time):
    """Using seconds since epoch, determine how much time has passed since the provided float. Returns string
    with hours:minutes:seconds"""
    elapsed_seconds = time() - start_time
    h = int(elapsed_seconds / 3600)
    m = int((elapsed_seconds - h * 3600) / 60)
    s = int((elapsed_seconds - m * 60) - h * 3600)
    return f"{h}hr {m}m {s}s"
