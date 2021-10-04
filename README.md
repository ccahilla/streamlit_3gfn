
[![DOI](https://zenodo.org/badge/388584835.svg)](https://zenodo.org/badge/latestdoi/388584835)

# streamlit_3gfn

Third generation frequency noise requirements and input optics design

## Link to interactive 3rd generation frequency noise budgets

[3rd generation frequency noise budgets](https://share.streamlit.io/ccahilla/streamlit_3gfn/main/streamlit_future_detector_freq_noise_budget.py)

## Instructions 

### How to reproduce Fig. 2 and Fig. 5 of 3rd generation frequency noise paper

0. Install Anaconda (https://www.anaconda.com/)

1. git clone this repository:

```
git clone https://github.com/ccahilla/streamlit_3gfn.git
```

2. Move to `streamlit_3gfn/code` directory:

```
cd streamlit_3gfn/code
```

3. Use the `environment.yml` in the `code` directory to produce the `3gfn` python3 environment:

```
conda env create --file environment.yml
```

4. Activate the environment

```
conda activate 3gfn
```

5. You should now be able to run the python code to produce Fig. 2 and Fig. 5 in the third generation frequency noise paper:

```
python current_detector_freq_noise_budget.py
```

and 

```
python future_detector_freq_noise_budget.py
```