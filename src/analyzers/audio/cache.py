"""Cache em JSON das respostas do Azure (ESTRATEGIA.md, seções 5 e 15).

Regra do time: nunca chamar de novo o que já está em cache, para não gastar a cota F0.
Reprocessar a mesma Consulta lê o JSON em vez de bater no Speech/Language de novo.

    uv run python -c "
    from pathlib import Path
    from src.analyzers.audio.cache import obter_ou_computar
    print(obter_ou_computar(Path('/tmp/c.json'), lambda: {'x': 1}))
    "
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

_IDENTIDADE: Callable[[Any], Any] = lambda v: v  # noqa: E731


def obter_ou_computar[T](
    caminho: Path,
    computar: Callable[[], T],
    *,
    para_json: Callable[[T], Any] = _IDENTIDADE,
    de_json: Callable[[Any], T] = _IDENTIDADE,
) -> T:
    """Lê `caminho` se existir; senão chama `computar()`, grava o resultado e o devolve.

    `para_json`/`de_json` convertem entre o tipo de `computar` (ex.: uma `list[dataclass]`)
    e algo serializável em JSON (ex.: `list[dict]`); por padrão não convertem nada.
    """
    if caminho.exists():
        return de_json(json.loads(caminho.read_text(encoding="utf-8")))

    resultado = computar()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(para_json(resultado), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resultado


def ja_em_cache(caminho: Path) -> bool:
    return caminho.exists()
