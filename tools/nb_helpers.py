"""Funções auxiliares para montar os notebooks do projeto via nbformat."""

import nbformat as nbf


def md(texto: str):
    return nbf.v4.new_markdown_cell(texto.strip() + "\n")


def code(texto: str):
    return nbf.v4.new_code_cell(texto.strip() + "\n")


def build_notebook(cells, caminho: str):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    with open(caminho, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook salvo: {caminho}")
