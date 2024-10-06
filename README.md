# Nlar

This project implements [brief description of your project] using Python. Below are the instructions for setting up the project, including how to configure the necessary shared path and run the project.

## Table of Contents

- [Installation](#installation)
- [Configuration Setup](#configuration-setup)
- [Running the Project](#running-the-project)
- [Contributing](#contributing)
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
Copy code
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

### Install Requirements
Once the virtual environment is activated, install the dependencies:

```bash
Copy code
pip install -r requirements.txt
```

## Configuration Setup

### Run Demo or Other Applications

If you only want to run the demo applications, no configuration is required. You can also simply
download the optimizers module where Nlarcm and Nlarsm are implemented. You can then simply
import these optimziers from the module.

### Run the Experiments

If you want to run the experiments of this project, it requires a simple configuration file to define the path to 
shared resources or other environment-specific settings. After cloning the repository, 
you need to create a configuration file based on the template provided.

- Option 1: Using a .py Configuration File
    - Go to the /config/ directory.
    - Copy config_template.py and rename the copy to config.py:

        Linux/macOS
        ``` bash
            cp config/config_template.py config/config.py
        ```

        Windows (Command Prompt or PowerShell)
        ```bash
           copy config\config_template.py config\config.py
        ```
    - Open the config.py file and update the SHARED_PATH variable with the appropriate path on your system:
``` python
    # config.py
    SHARED_PATH = '/your/local/path/to/shared/resource'
```

- Option 2: Using a .txt or .env Configuration File
    - Go to the /config/ directory.
    - Copy config_template.txt (or .env.template) and rename it to config.txt (or .env):

        Linux/macOS
        ```bash
        cp config/config_template.txt config/config.txt
        ```

        Windows (Command Prompt or PowerShell)
        ```bash
        copy config\config_template.txt config\config.txt
        ```
    - Open the config.txt (or .env) file and update the SHARED_PATH with your local path:
        ``` txt
        SHARED_PATH=/your/local/path/to/shared/resource
        ```

        Note: If the path contains spaces, you can wrap it in quotes like so:
        SHARED_PATH='/your/local/path with spaces/resource'.