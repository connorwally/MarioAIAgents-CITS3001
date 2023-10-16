# CITS3001-Project

A project for the CITS3001 unit implementing 2 agents to play Super Mario Bros.

Project created by:

- Reilly Evans (23615971)
- Connor Wallis (22506057)

# Installation

When installing and running the project, you may encounter multiple errors so please follow the installation guide closely:

1. Ensure the following:
   * A Python version <3.12 is installed. (Our project was run on 3.11.5)
   * Poetry is installed (see https://python-poetry.org/docs/master/)
   * Poetry is added to your system path correctly (see above link)
   * Visual Studio BuildTools is installed
   * For the PPO agent, you are using Windows (does not work in MacOS)
2. Change directory into the project file using the terminal command “cd [file path]”
3. Install the required dependencies through poetry using the command “poetry install”
4. Running:
   a. Rule-based Agent: “poetry run python ruleBasedMario.py”
   b. Train the PPO Agent: “poetry run python 1_TrainMario.py”
   c. Run the PPO Agent: “poetry run python 2_RunMario.py”
   d. Run the PPO Deterministic Agent: "poetry run python 3_RunMarioDeterministic.py"

NOTE:
- The default agent model is our best model to date. Using a learning rate of 0.00001 and running on super-mario-bros-gym-v3.
- This will install the PyTorch CPU version (GPU/CUDA version requires some work to install through Poetry) and thus the program may execute slowly.

# PPO Models
For the project we developed over 50 different model configurations and hundreds of models (although most were short lived). Included is only 3, to keep the file space small.
Change the variable called MODEL_PATH in 2_RunMario.py or 3_RunMarioDeterministic.py to one of the below:
1. "./models/model_best_v3_highlearningrate.zip" - Trained on a learning rate of 0.00001 and the v3 of mario.
2. "./models/model_medium_v1_lowlearningrate.zip" - Trained on a learning rate of 0.000001 and the v0 of mario.
3. "./models/model_bad_v1_highlearningrate.zip" - Trained on a learning rate of 0.00001 and the v0 of mario.
