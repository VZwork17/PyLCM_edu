# PyLCM_parcel
Lagrangian Cloud Model(LCM) parcel model for education purposes. 

> **Note**: This is the `perturbed-physics-modifications` branch. See changes below.

## Branch: perturbed-physics-modifications

This branch contains modified code to run different versions of the model (turbulent ascent and decoupled microphysics). The main changes are:

### Turbulent ascent
- **ascend_parcel()** now includes a turbulent and white noise mode
- **timestep_routine.py** has been changed to accommodate the additional input parameters for turbulent ascending modes
- **aero_init.py** initializes `wp_parcel` for turbulent ascending modes
### Decoupled microphysics
- this is solved with an additional module `src/physics/condensation_moist_adiabat.py`

Additionally, now the model builds (piece-wise) environment profiles from height-value tuples with `compute_environment_profiles()` as alternative to the presets in `create_env_profiles()`


## Installation
1. install anaconda3 for python / jupyter
https://docs.anaconda.com/

2. Create a conda environment for PyLCM (Mac/Linux)
```
conda create -n PyLCM
```
3. Activate conda env (Mac/Linux).
```
conda activate  PyLCM
```

4. Install necessary packages (jupyter,numpy, scipy, pandas, matplotlib, plotly, ipywidgets)

#For Mac/Linux users:
```
  conda install jupyter
  conda install -c plotly plotly
  conda install numpy
  conda install scipy
  conda install pandas
  conda install matplotlib
  conda install ipywidgets
```

#For Windows users:
  Please install the packages listed above via Anaconda Navigator

## Model Run
To run the PyLCM model, follow these steps:
1. Activate conda env (Mac/Linux).
```
conda activate  PyLCM
```
2. run jupyter
```
   jupyter notebook
```
3. run `PyLCM_edu.ipynb' from your browser 


## Usage
- The Python files in the PyLCM module are core files for running the model. It is recommended only to modify them if you fully understand their functionality.
- For post-processing, useful codes are included in the `Post_process` module, such as `analysis.py` and `print_plot.py.` Users can use or modify these codes to create the desired plots.

* Contact: J.lim@physik.uni-muenchen.de
