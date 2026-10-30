# cshl-qb-course

## Setup instructions

0. Clone this repository:
   `git clone git@github.com:Hou-Lab-CSHL/cshl-qb-course.git`
1. Change directories into the project folder: `cd path/to/your/project`
   (e.g. `cd /Users/hou/Documents/GitHub/cshl-qb-course/`)
2. Install the packages listed in `pyproject.toml` (requires Python 3.11).
   Any package manager works. We recommend
   [uv](https://docs.astral.sh/uv/), which also installs Python for you.

   **Option A: uv (recommended)**
   - Install uv
     ([instructions](https://docs.astral.sh/uv/getting-started/installation/))
   - Install Python 3.11 and all packages into `.venv/`: `uv sync`
   - Start Jupyter: `uv run jupyter lab`

   **Option B: pip**
   - Install Python 3.11 (e.g. from [python.org](https://www.python.org/))
   - Create an environment: `python3.11 -m venv .venv`
   - Activate it: `source .venv/bin/activate`
     (Windows: `.venv\Scripts\activate`)
   - Install all packages: `pip install -e .`
   - Start Jupyter: `jupyter lab`

   You can also open the notebooks in VS Code and select the `.venv` kernel.
3. Download the data (follow the link in your email) and unzip the
   containing data folders directly into your `/cshl-qb-course/` directory
   (`/cshl-qb-course/fe-data/`, `/cshl-qb-course/ephys-data/` etc.)

During class, do a `git pull` in your `/cshl-qb-course/` directory to fetch
the latest content, then reinstall to pick up any new packages (`uv sync`
or `pip install -e .`).
