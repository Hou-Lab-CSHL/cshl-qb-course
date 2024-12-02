# cshl-qb-course

## Setup instructions

1. Change directories to the project folder: `cd path/to/your/project`
2. Download and install Micromamba (recommended though Conda, etc. should work):
   https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#automatic-install
3. Create the environment: `micromamba create -f environment.yaml`
4. Activate the environment: `micromamba activate cshl-qb-course`
5. Install packages: `poetry install --no-root`