<a id="readme-top"></a>

<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains two Jupyter Notebook files, `spin_orbit_coupling_2.ipynb` and `g_factor.ipynb`, that can be used for interactive data analysis. They allow semi-automated readout of the spin-orbit coupling energy and the spin- and valley-coupling factors.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

I recommend using [uv](https://docs.astral.sh/uv/) for downloading the required packages in Python. Follow the [steps in the documentation to install uv](https://docs.astral.sh/uv/getting-started/installation/). On Windows, the command listed in the docs needs to be run using PowerShell (and not the Command Prompt). You will need to restart PowerShell and VS Code after the installation for the command `uv` to work.

Alternatively, you can just use your own virtual environment and download any packages you might still need with pip or conda.

### Installation

1. Inside PowerShell, navigate to a folder where you would like to download the code using the command `cd`.

2. Clone the repository with the command
   ```sh
   git clone https://github.com/nanophysics/interactive-analysis-tools.git
   ```
3. Go into the new folder with
   ```sh
   cd interactive-analysis-tools
   ```
4. Use uv to create a new venv with all the required packages
   ```js
   uv sync
   ```
5. Inside VS Code or Jupyter Notebook, select the kernel called `Code2`, which should be located in `.venv\Scripts\python.exe`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>