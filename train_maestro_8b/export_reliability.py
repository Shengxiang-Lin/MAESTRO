"""Legacy CLI compatibility. Use python -m train_maestro_8b.data.export_risk."""
from train_maestro_8b.data.export_risk import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
