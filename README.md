# cs269

Install dependencies with `pip install -e .`

Test the environment with `make test`. Run RL training with `make train`. Run testing of the agent with `make agent`. Run training and testing of the agent with `make all`.

If running on MacOS, install mjpython and replace `PYTHON = python3` in the makefile with `PYTHON = mjpython`

For `make manual_control` use arrow keys to go left and right, `:` and `>` keys to go up and down, and space bar to grab and release the gripper

The python scripts inside the results folder were used to parse the outputs from our training and test time evaluation and plot them as graphs