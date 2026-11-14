"""Legacy CLI compatibility. Use python -m train_maestro_8b.data.export_policy."""
from train_maestro_8b.data.export_policy import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
