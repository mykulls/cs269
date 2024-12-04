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

all: train agent