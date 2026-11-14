"""Legacy CLI compatibility. Use python -m train_maestro_8b.policy.train."""
from train_maestro_8b.policy.train import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
