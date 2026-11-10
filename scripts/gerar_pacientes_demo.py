"""Grava no SQLite os Alertas dos Pacientes de Demonstração, sem precisar tocar o replay.

    uv run python -m scripts.gerar_pacientes_demo

Tira os Alertas mock do banco e preserva Reconhecimentos; reexecutar não duplica nada.
"""

from __future__ import annotations

from src.core import db, demo


def main() -> None:
    with db.conectar() as conn:
        for paciente in demo.carregar_pacientes():
            video = demo.alertas_de_video(conn, paciente.id)
            eventos = demo.fundir_eventos(demo.montar(paciente).eventos, video)
            novos = demo.gravar(conn, (evento.alerta for evento in eventos))
            print(f"{paciente.id}: {len(eventos)} Alertas, {novos} novos")


if __name__ == "__main__":
    main()
