"""Compatibility entry point for the DeBERTa-v3 Task 2 baseline.

The complete implementation is kept in ``train_deberta_v3.py``.  This small
wrapper provides the exact filename requested by the group-project handover.
"""

from train_deberta_v3 import main


if __name__ == "__main__":
    main()
