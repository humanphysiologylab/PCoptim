# PCoptim

This distribution contains code for processing patch-clamp data via experimetnal setup mnodel optimization. Details are in the article 'Human sodium current voltage-dependence at physiological temperature measured by coupling patch-clamp experiment to a mathematical model.'

## Installation

1. This code uses Sundials CVODE solver v.6.5.1. Cmake v.3.8 or above is used to liki model libraries.

2. Compile the patch-clamp model C code. 
```
cd ./src/model_ctypes
cmake .
make
cd ../..
````

3. Install necessary python libraries
```
pip install numpy pandas matplotlib mpi4py tqdm pyafb
```
### Conda environment
Conda users might want to create virtual environment from the ina_env.txt instead.
```
conda create --prefix ./ina_env --file ina_env.txt
```

Activate conda enviroment

```
conda activate ina_env/
```
Install conda env as jupyter kernel.
```
python -m ipykernel install --user --name=ina_env
```

4. Install pypoptim library from https://github.com/humanphysiologylab/pypoptimin SOME_DIR_TO_PYPOPTIM

```
git clone https://github.com/humanphysiologylab/pypoptim.git
```

go to SOME_DIR_TO_PYPOPTIM and install the library 
    cd SOME_DIR_TO_PYPOPTIM/pypoptim
    pip install .

## Usage
Run genetic algorithm on N threads with JSON.json config from PCoptim directory
...
mpirun -n N python3 ./ga/mpi_scripts/mpi_script.py ./ga/configs/JSON.json
...

## Results
Results will be saved  in ./results

## Notebooks
1. [001_Test_ina_model_ctypes.ipynb](./notebooks/001_Test_ina_model_ctypes.ipynb)
In this notebook presented, how can be look like model of current. It requires library.so  for counting 
and .ga/mpi_scripts/ina_model.py for working with input data(such as list of constants, protocols and etc.)
and returning output current. 
2. 002_Patch_clamp_output_files_preprocessing_for_ga.ipynb

Test model oprimization with a single thread
```
python3 ./ga/mpi_scripts/mpi_script.py ./ga/configs/test.json 
```

Test model oprimization with 2 threads using  Open MPI.
```
mpirun -n 2 python3 ./ga/mpi_scripts/mpi_script.py ./ga/configs/test.json 
```


## Authors
Veronica Abrasheva and Roman Syunyaev