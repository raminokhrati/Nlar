# Nlar

## Title

Dynamic Estimation of Learning Rates Using a Non-Linear Autoregressive Model

## Abstract

We introduce a new class of adaptive non-linear autoregressive (Nlar) models incorporating the concept of momentum, which dynamically estimate both the learning rates and momentum as the number of iterations increases. In our method, the growth of the gradients is controlled using a scaling (clipping) function, leading to stable convergence. Within this framework, we propose three distinct estimators for learning rates and provide theoretical proof of their convergence. We further demonstrate how these estimators underpin the development of effective Nlar optimizers. The performance of the proposed estimators and optimizers is rigorously evaluated through extensive experiments across several datasets and a reinforcement learning environment. The results highlight two key features of the Nlar optimizers: robust convergence despite variations in underlying parameters, including large initial learning rates, and strong adaptability with rapid convergence during the initial epochs.

This project implements Nlar algorithm using Python. For more details, please check out the full paper on arXiv: https://arxiv.org/abs/2410.09943

Below are the instructions for setting up the project, 
including how to configure the necessary shared path and run the project.

## Table of Contents

- [Installation](#installation)
- [Configuration Setup](#configuration-setup)
  - [Run Demo or Other Applications](#run-demo-or-other-applications)
  - [Run the Experiments](#run-the-experiments)
- [License](#license)

---

## Installation

To run this project, you need to have Python installed. The recommended version is Python 3.7+.

### 1. Clone the Repository

```bash
git clone https://github.com/raminokhrati/Nlar.git
cd Nlar
```

### Install Dependencies
It is recommended to use a virtual environment for dependency management. You can set up a virtual environment and install the required dependencies as follows:

Linux/macOS
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

Windows
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

### Install Requirements
Once the virtual environment is activated, install the dependencies:

```bash
pip install -r requirements.txt
```

## Configuration Setup

### Run Demo or Other Applications

If you only want to run the demo applications, no configuration is required. You can simply
download the optimizers module where Nlarcm and Nlarsm are implemented and import these optimziers from the module.

### Run the Experiments

If you want to run the experiments of this project, it requires a simple configuration file to define the path to 
shared resources or other environment-specific settings. After cloning the repository, 
you need to create a configuration file based on the template provided.

- Option 1: Using a .py Configuration File
    - Go to the experiments/config/ directory.
    - Copy config_template.py and rename the copy to config.py:

        Linux/macOS
        ``` bash
            cp experiments/config/config_template.py experiments/config/config.py
        ```

        Windows (Command Prompt or PowerShell)
        ```bash
           copy experiments\config\config_template.py experiments\config\config.py
        ```
    - Open the config.py file and update the shared_path variable with the appropriate path on your system:
``` python
    # config.py
    shared_path = '/..../Nlar' # your local path to shared resource assuming that the project is saved in a folder named Nlar
```

- Option 2: Using a .txt or .env Configuration File
    - Go to the experiments/config/ directory.
    - Copy config_template.txt (or .env.template) and rename it to config.txt (or .env) by running the following command (you may need to refresh the directory to see the effect):

        Linux/macOS
        ```bash
        cp experiments/config/config_template.txt experiments/config/config.txt
        ```

        Windows (Command Prompt or PowerShell)
        ```bash
        copy experiments\config\config_template.txt experiments\config\config.txt
        ```
    - Open the config.txt (or .env) file and update the shared_path with your local path:
        ``` txt
        shared_path = '/.../Nlar' # your local path to shared resource assuming that the project is saved in a folder named Nlar
        ```

        Note: If the path contains spaces, you can wrap it in quotes like so:
        SHARED_PATH='/your/local/path with spaces/resource'.

**Once the right path is configured, the experiments can be run by executing "run_experiments.py" file in the folder "experiments".
Alternatively, they can be run individually from their python files in their respective folders:
folder "classifications"
for classification experiments, folder "reinforcement-learning-ddqn" for the reinforcement learning experiment, and folder "sensitivity"
for sensitivity experiments, all located in folder "experiments".**

## License

This project is licensed under the MIT License.
