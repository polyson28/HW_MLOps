import sys
from pathlib import Path

# pytest запускает тесты так что корень проекта не всегда попадает в sys.path.
# из-за этого импорт вида from app... может не находиться
# потэому руками добавляем корневую папку проекта в path на время тестов
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
