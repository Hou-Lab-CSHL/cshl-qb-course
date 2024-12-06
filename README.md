# cshl-qb-course

## Setup instructions

0. Clone this repository: `git clone git@github.com:Hou-Lab-CSHL/cshl-qb-course.git`
1. Change directories into the project folder: `cd path/to/your/project` (e.g. `cd /Users/hou/Documents/GitHub/cshl-qb-course/`)
2. Download and install Micromamba (recommended although Conda, etc. should work):
   https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#automatic-install
3. Create the environment: `micromamba create -f environment.yaml`
4. Activate the environment: `micromamba activate cshl-qb-course`
5. Install packages: `poetry install --no-root`
6. Download the data (follow the dropbox link in your email) and unzip the containing data folders directly into your `/cshl-qb-course/` directory (`/cshl-qb-course/fe-data/`, `/cshl-qb-course/ephys-data/` etc.)

During class, do a `git pull` in your `/cshl-qb-course/` directory to fetch the latest content. 
