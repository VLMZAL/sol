# Sol

Sol is an experimental Python project that uses a small quantum neural network built with PennyLane to control an agent in a simple 2D navigation environment rendered with Pygame.

## Installation

Requires Python 3. Create and activate a virtual environment on macOS or Linux, then install the dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train or update the weights and save them to `weights.npy`:

```sh
python train.py
```

Run the Pygame visualization using the saved weights:

```sh
python main.py
```

`weights.npy` is generated locally and is not tracked by Git. Run the training script before the visualization if no saved weights exist.

## Project structure

- `env.py` defines the 2D navigation environment and reward.
- `qnn.py` defines the PennyLane quantum circuit and policy output.
- `train.py` updates and saves the weights.
- `main.py` runs the Pygame visualization.
