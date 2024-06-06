# card_recognition_ml
Identifies cards based on an image. 


# to activate the venv
    .venv\Scripts\activate
    source venv/Scripts/activate
# do deactivate the venv
deactivate




# Packages to Install (in .venv)
pip install tensorflow matplotlib pyyaml h5py
pip install scikit-learn opencv-python

pip install scikit-learn opencv-python keras numpy pandas Pillow requests tensorflow


 
# To Run on a Server
0.1) make the virtual environment
NOTE: if the .venv is already made, then you don't have to do this again
'python3 -m venv .venv'

0.2) activate the virtual environment (fron inside the directory)
NOTE: you must do this before installing or running anything
'./.venv/Scripts/activate'

0.3) install the nessicary packages
NOTE: you can skip this after installing them once
'pip install scikit-learn opencv-python keras numpy pandas Pillow requests tensorflow matplotlib tensorflow_probability tensorflow-probability[tf]'

0.4) deactivate the .venv
NOTE: you must have first activated the .venv
'deactivate'

1.0) make sure the dataset is inside the .data file (the json files will be stored here)

2.0) run the program
NOTE: make sure that you have activated the virtual environment
'python3 main_pray.py'


