# PCoptim

This distribution contains code for processing patch-clamp data via experimental setup model optimization. Details are in the article 'Human sodium current voltage-dependence at physiological temperature measured by coupling patch-clamp experiment to a mathematical model.'

## Installation

1. This code uses Sundials CVODE solver v.6.5.1. Cmake v.3.8 or above is used to link the libraries.

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
#### Conda environment (optional)
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

4. Install pypoptim library from https://github.com/humanphysiologylab/pypoptimin PYPOPTIM_DIR

```
git clone https://github.com/humanphysiologylab/pypoptim.git
```


    cd PYPOPTIM_DIR/pypoptim
    pip install .

## Usage
Run genetic algorithm on N threads with JSON.json config from PCoptim directory
...
mpirun -n N python3 ./ga/mpi_scripts/mpi_script.py ./ga/configs/JSON.json
...
## Test
Test model oprimization with a single thread using 'test.json' configuration file.
```
python3 ./ga/mpi_scripts/mpi_script.py ./ga/configs/test.json 
```
Test model oprimization with 2 threads using  Open MPI.
```
mpirun -n 2 python3 ./ga/mpi_scripts/mpi_script.py ./ga/configs/test.json 
```

## Results
Results will be saved  in ./results

## Example jupyter-notebooks
[001_Test_ina_model_ctypes.ipynb](./notebooks/001_Test_ina_model_ctypes.ipynb)

Example to test patch clamp model. Sodium current voltage-clamp model used in original publication is compiled as libina.so C library. Custom user-defined models can be used, user should provide 'run' and 'status' methods. Details are given in the notebook example file.

[002_Patch_clamp_output_files_preprocessing_for_ga.ipynb](./notebooks/002_Patch_clamp_output_files_preprocessing_for_ga.ipynb)

Example importing patch-clamp data from axon binary or text files (file.abf, file.atf).

[003_Make_weight_for_trace.ipynb](./notebooks/003_Make_weight_for_trace.ipynb)

Example notebook to generate weights for trace

[004_Create_json_file.ipynb](./notebooks/004_Create_json_file.ipynb)

Example script to generate .json configuration file required for GA optimization.

[005_Results_processing.ipynb](./notebooks/005_Results_processing.ipynb)

Example notebook how to load ga results

## Authors
Veronica Abrasheva and Roman Syunyaev
