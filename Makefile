# Define the Python interpreter
PYTHON = python3

# Rule to train the agent
train:
	$(PYTHON) scripts/train_agent.py

# Rule to test the environment
test:
	$(PYTHON) scripts/test_env.py

# Rule to train and test the agent
agent:
	$(PYTHON) scripts/test_agent.py

hard_coded_action:
	$(PYTHON) scripts/hard_code_flip.py

manual_control:
	$(PYTHON) scripts/manual_control_robot.py

all: train agent