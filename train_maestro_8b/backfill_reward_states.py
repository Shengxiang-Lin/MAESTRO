"""Legacy CLI compatibility. Use python -m train_maestro_8b.data.backfill_rewards."""
from train_maestro_8b.data.backfill_rewards import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
