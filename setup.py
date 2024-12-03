from setuptools import setup, find_packages

setup(
    name="bottle_flip_envs",
    version="0.1",
    description="Custom bottle flip environment built on Robosuite",
    author="Michael Shi",
    author_email="michshi8@gmail.com",
    packages=find_packages(),  # Automatically find all packages in your repository
    install_requires=[
        "robosuite>=1.5.0",   # Robosuite package
        "robosuite_models",    # Include robosuite_models package
        "gym>=0.21.0",        # Gym for RL compatibility
        "stable-baselines3>=1.7.0"  # Stable-Baselines3 for training
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'train_agent = scripts.train_agent:main',  # If you have a train_agent script
        ],
    },
)