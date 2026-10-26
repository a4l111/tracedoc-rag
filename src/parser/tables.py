"""Render extracted tables without losing merged-cell structure."""

from html import escape


def markdown_table(rows: list[list[str]]) -> str:
    """Render a rectangular table with its first row as the header."""
    if not rows:
        return ""
    width = max(len(row) for row in rows)

    def cells(row: list[str]) -> str:
        values = [(value or "").replace("|", r"\|").replace("\n", "<br>") for value in row]
        return "| " + " | ".join(values + [""] * (width - len(values))) + " |"

    return "\n".join([cells(rows[0]), "| " + " | ".join(["---"] * width) + " |",
                      *(cells(row) for row in rows[1:])])


def docx_table(rows: list[list[dict[str, object]]]) -> str:
    """Use Markdown for plain cells and HTML for horizontal/vertical merges."""
    if not rows:
        return ""
    merged = any(cell["colspan"] != 1 or cell["vmerge"] is not None for row in rows for cell in row)
    if not merged:
        return markdown_table([[str(cell["text"]) for cell in row] for row in rows])

    active: dict[int, dict[str, object]] = {}
    placed: list[list[dict[str, object]]] = []
    for row in rows:
        output_row = []
        column = 0
        for cell in row:
            colspan = int(cell["colspan"])
            if cell["vmerge"] == "continue":
                anchor = active.get(column)
                if anchor is not None:
                    anchor["rowspan"] = int(anchor["rowspan"]) + 1
                column += colspan
                continue
            rendered = {"text": cell["text"], "colspan": colspan, "rowspan": 1}
            output_row.append(rendered)
            for offset in range(colspan):
                if cell["vmerge"] == "restart":
                    active[column + offset] = rendered
                else:
                    active.pop(column + offset, None)
            column += colspan
        placed.append(output_row)

    lines = ["<table>"]
    for row in placed:
        lines.append("<tr>")
        for cell in row:
            attributes = ""
            if cell["colspan"] != 1:
                attributes += f' colspan="{cell["colspan"]}"'
            if cell["rowspan"] != 1:
                attributes += f' rowspan="{cell["rowspan"]}"'
            lines.append(f"<td{attributes}>{escape(str(cell['text']))}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)
