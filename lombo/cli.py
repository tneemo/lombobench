"""CLI do LomboBench: `python -m lombo`."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from .proxy import LomboConfig, create_app

PRIVACY_NOTICE = (
    "AVISO: o LomboProxy grava prompts e respostas interceptados em um banco local.\n"
    "As conversas podem conter dados sensíveis. Use X-Lombo-No-Log para excluir uma\n"
    "requisição do registro, ou --no-capture para desativar a captura."
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lombo",
        description="LomboBench — proxy transparente + Lab para observar os "
                    "controles externos de inferência do Ollama.")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--host", default="127.0.0.1",
                        help="por privacidade, o padrão escuta apenas localhost")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--db", default="lombo.db")
    parser.add_argument("--no-capture", action="store_true",
                        help="repassa as requisições sem gravar nada")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.host != "127.0.0.1":
        print(f"ATENÇÃO: escutando em {args.host} — o proxy fica acessível fora desta "
              f"máquina e as conversas capturadas podem conter dados sensíveis.")
    print(PRIVACY_NOTICE if not args.no_capture else "Captura DESATIVADA (--no-capture).")
    print(f"\n  Proxy (drop-in do Ollama): http://{args.host}:{args.port}")
    print(f"  Lab / análise:             http://{args.host}:{args.port}/_lombo/")
    print(f"  Ollama upstream:           {args.ollama_url}")
    print(f"  Banco:                     {args.db}\n")

    config = LomboConfig(ollama_url=args.ollama_url, db_path=args.db,
                         capture=not args.no_capture)
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
