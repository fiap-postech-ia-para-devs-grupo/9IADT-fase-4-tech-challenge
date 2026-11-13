"""Isola os testes do banco real.

Telas do Streamlit abrem `db.conectar()` sem caminho, que usa `data/alertas.db`. O caminho
padrão é lido de `ALERTAS_DB` quando `src.core.db` é importado, então ele é trocado aqui,
antes de qualquer teste importar o pacote.
"""

import os
import tempfile
from pathlib import Path

os.environ["ALERTAS_DB"] = str(Path(tempfile.mkdtemp(prefix="alertas-teste-")) / "alertas.db")
