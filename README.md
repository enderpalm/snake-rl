## Snake RL

Reinforcement Learning (RL) single-agent environment for the classic Snake game, implemented using the Gymnasium API. Featuring partial and full grid observations, multiple reward shaping options, and a simple Pygame-based UI for human interaction

### Folder Structure

The project is organized as follows:

```text
snake-rl/
├── agents/          :: Agent implementations 
├── artifacts/       :: Saved models and metrics
├── core/            :: Core environment logic (gym_env.py, Pygame UI, enums)
├── notebooks/       :: Jupyter notebooks for training and experiments
├── resources/       
├── scripts/         :: Executable run scripts for training and testing agents
├── pyproject.toml   :: Project metadata and dependencies
└── uv.lock          
```

### How to Setup

This project uses `uv` as its package and environment manager.

1. **Install dependencies and sync the environment**
   Run `uv sync` to automatically read the `pyproject.toml` and `uv.lock` files, create a virtual environment, and install all dependencies.

    ```bash
    uv sync
    ```

2. **Activate the virtual environment**
   Once installed, you can activate the environment created by `uv`:
    ```bash
    source .venv/bin/activate
    ```
    > (For Windows: `.venv\Scripts\activate`)

### Running the Code

Once the environment is properly set up, you can run scripts and interact with agents. For example, testing the q-learning

```bash
uv run python -m scripts.test_q_learning
```

### UI Controls

When running the environment with the `HUMAN` render mode, you can interact with the Pygame window using the following controls:

- **Spacebar**: Pause or unpause the simulation.
- **Left Click**: When paused, place an obstacle on the grid.
- **Right Click**: When paused, place an apple on the grid.

---
