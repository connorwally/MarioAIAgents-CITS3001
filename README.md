# CITS3001-Project

A project for the CITS3001 unit implementing 2 agents to play Super Mario Bros.

Project created by:

- Reilly Evans (23615971)
- Connor Wallis (22506057)

# Installation

When installing and running the project, you may encounter multiple errors so please follow the installation guide closely:

1. Ensure the following:
   a. A Python version <3.12 is installed. (Our project was run on 3.11.5)
   b. Poetry is installed (see https://python-poetry.org/docs/master/)
   c. Poetry is added to your system path correctly (see above link)
   d. Visual Studio BuildTools is installed
   e. For the PPO agent, you are using Windows (does not work in MacOS)
2. Change directory into the project file using the terminal command “cd [file path]”
3. Install the required dependencies through poetry using the command “poetry install”
4. Running:
   a. Rule-based Agent: “poetry run python ruleBasedMario.py”
   b. Train the PPO Agent: “poetry run python 1_TrainMario.py”
   c. Run the PPO Agent: “poetry run python 2_RunMario.py”

NOTE:
- The default agent model is our best model to date. Using a learning rate of 0.00001 and running on super-mario-bros-gym-v3.
- This will install the PyTorch CPU version (GPU/CUDA version requires some work to install through Poetry) and thus the program may execute slowly.
