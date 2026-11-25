"""The History tab (``history.History``): find, label, recover and reuse every take.

* a table of takes (newest first) with search and filters (kind, model, starred, deleted);
* picking a take (by its id) plays it and shows what was asked for, plus an edit panel: title, tags, notes, star;
* buttons: **Open in its tab** (re-fills the tab the take came from, so you can change it and sing again),
  **Delete** / **Restore** (soft delete: files go to ``outputs/trash/``), **Export** (zip of the chosen takes);
* the recent event log (what the app did, cache hits, errors) and **Rebuild history** (re-creates ``history.jsonl``
  from the per-take ``.json`` files if it is ever lost).

``build_history_tab`` returns the components the app needs to wire "Open in its tab" (``open_btn``, ``rid``) and to
refresh the table after new takes (``refresh``).
"""
from __future__ import annotations

import json
from typing import Dict, List

from .history import describe, quality, what

COLS = ["★", "id", "time", "kind", "model", "voice", "what", "title / tags", "quality"]
KINDS = ["any", "sing", "resing", "sargam", "compose"]


def rows(hist, text="", kind="any", model="any", starred=False, deleted=False) -> List[List]:
    """Table rows for the filters."""
    out = []
    for r in hist.search(text, "" if kind == "any" else kind, "" if model == "any" else model, starred, deleted):
        q = r.get("request", {})
        voice = q.get("voice") or (q.get("comp") or {}).get("voice") or ""
        tt = " ".join(x for x in [r.get("title") or "", " ".join(f"#{t}" for t in (r.get("tags") or []))] if x)
        out.append(["★" if r.get("star") else "", r["id"], r["time"], r["kind"], r["model"], voice, what(r)[:60], tt,
                    quality(r)])
    return out


def details(r: Dict) -> str:
    """Markdown: what was asked for and how it went."""
    info = {k: v for k, v in r.get("info", {}).items() if k not in ("sung", "lines")}
    lines = [f"**#{r['id']}** {r['time']} · {r['kind']} · {r['model']}" + (" · deleted" if r.get("deleted") else ""),
             f"`{r['wav']}`"]
    if r.get("info", {}).get("lines"):
        lines.append("\n".join(f"- {l.get('line')} (tune {l.get('tune')}, voice {l.get('voice')})"
                               for l in r["info"]["lines"]))
    lines.append("Request:\n```json\n" + json.dumps(r.get("request", {}), ensure_ascii=False, indent=1)[:3000] + "\n```")
    lines.append("Info:\n```json\n" + json.dumps(info, ensure_ascii=False, indent=1)[:2000] + "\n```")
    return "\n\n".join(lines)


def events_md(hist, n: int = 15) -> str:
    """The last events as a small table."""
    ev = hist.events(n)
    if not ev:
        return "No events yet."
    body = []
    for e in ev:
        extra = {k: v for k, v in e.items() if k not in ("time", "event", "request", "trace")}
        txt = json.dumps(extra, ensure_ascii=False)[:140].replace("|", "/")
        body.append(f"| {e['time']} | {e['event']} | {txt} |")
    return "| time | event | details |\n|---|---|---|\n" + "\n".join(body)


def build_history_tab(gr, hist):
    """The History tab (call inside ``gr.Blocks``). Returns {open_btn, rid, refresh, table, filters}."""
    with gr.Tab("History"):
        gr.Markdown(f"Every take is kept in `{hist.dir}` (wav + json), logged in `history.jsonl`, labelled in "
                    "`annotations.jsonl`; everything the app does goes to `log.jsonl`. Type an id (or click a row) "
                    "to play a take.")
        with gr.Row():
            text = gr.Textbox(label="Search (words, title, notes, tags)", scale=3)
            kind = gr.Dropdown(KINDS, value="any", label="Kind", scale=1)
            model = gr.Dropdown(["any", "kalavinka", "base"], value="any", label="Model", scale=1)
            star = gr.Checkbox(label="★ only", scale=1)
            deleted = gr.Checkbox(label="Deleted ones", scale=1)
            refresh = gr.Button("Refresh", scale=1)
        table = gr.Dataframe(rows(hist), headers=COLS, interactive=False, wrap=True, max_height=360)
        with gr.Row():
            rid = gr.Textbox(value="", label="Take id", placeholder="e.g. 12 (or click a row)", scale=1)
            audio = gr.Audio(label="Take", type="filepath", interactive=False, scale=3)
        info = gr.Markdown()
        with gr.Accordion("Label this take", open=True):
            with gr.Row():
                title = gr.Textbox(label="Title", scale=2)
                tags = gr.Textbox(label="Tags (space or comma separated)", scale=2)
                starred = gr.Checkbox(label="★ Star", scale=1)
            notes = gr.Textbox(label="Notes", lines=2)
            with gr.Row():
                save = gr.Button("Save labels", variant="primary")
                open_btn = gr.Button("Open in its tab")
                delete = gr.Button("Delete")
                restore = gr.Button("Restore")
                export = gr.Button("Export shown takes (zip)")
            status = gr.Markdown()
            zipf = gr.File(label="Export", interactive=False)
        with gr.Accordion("What happened (event log)", open=False):
            log = gr.Markdown(events_md(hist))
            rebuild = gr.Button("Rebuild history from the take files")
        filters = [text, kind, model, star, deleted]

        def do_rows(*f):
            return rows(hist, *f), events_md(hist)
        for f in filters:
            f.change(do_rows, filters, [table, log])
        refresh.click(do_rows, filters, [table, log])

        def do_select(data, evt):
            try:
                r = evt.index[0]
                return str(int(data.iloc[r, 1]) if hasattr(data, "iloc") else int(data[r][1]))
            except Exception:
                return gr.update()
        # Gradio injects the click event by the parameter's type; this module uses postponed (string)
        # annotations, so the real type is attached here
        do_select.__annotations__ = {"data": object, "evt": gr.SelectData}
        table.select(do_select, table, rid)

        def do_show(n):
            n = str(n or "").strip().lstrip("#")
            if not n.isdigit():
                return None, "", "", "", False, ""
            r = hist.get(int(n), full=True)
            if r is None:
                return None, f"No take {int(n)}.", "", "", False, ""
            path = hist.wav_path(r)
            return (str(path) if path.exists() else None, details(r), r.get("title") or "",
                    " ".join(r.get("tags") or []), bool(r.get("star")), r.get("notes") or "")
        rid.change(do_show, rid, [audio, info, title, tags, starred, notes])

        def need(n):
            n = str(n or "").strip().lstrip("#")
            if not n.isdigit() or hist.get(int(n)) is None:
                raise gr.Error("pick a take first (type its id or click a row)")
            return int(n)

        def do_save(n, t, tg, st, nt, *f):
            n = need(n)
            old = hist.get(n)
            for field, val in (("title", t.strip()), ("tags", tg), ("star", bool(st)), ("notes", nt.strip())):
                cur = old.get(field)
                if field == "tags":
                    cur = " ".join(cur or [])
                if (cur or "") != (val or "") and not (field == "star" and bool(cur) == val):
                    hist.annotate(n, field, val)
            return f"Saved labels of #{n}.", *do_rows(*f)
        save.click(do_save, [rid, title, tags, starred, notes] + filters, [status, table, log])

        def do_delete(n, *f):
            n = need(n)
            hist.delete(n)
            return f"#{n} moved to the trash (tick 'Deleted ones' to see it, then Restore).", *do_rows(*f)
        delete.click(do_delete, [rid] + filters, [status, table, log])

        def do_restore(n, *f):
            n = need(n)
            hist.restore(n)
            return f"#{n} restored.", *do_rows(*f)
        restore.click(do_restore, [rid] + filters, [status, table, log])

        def do_export(*f):
            ids = [r[1] for r in rows(hist, *f)]
            if not ids:
                raise gr.Error("no takes to export")
            return str(hist.export(ids)), f"Exported {len(ids)} takes."
        export.click(do_export, filters, [zipf, status])

        def do_rebuild(*f):
            n = hist.rebuild()
            return f"Rebuilt history.jsonl from {n} take files (the old one is kept as a .bak file).", *do_rows(*f)
        rebuild.click(do_rebuild, filters, [status, table, log])
    return {"open_btn": open_btn, "rid": rid, "refresh": refresh, "table": table, "filters": filters,
            "status": status, "rows": lambda *f: rows(hist, *f), "log": log}


__all__ = ["build_history_tab", "rows", "details", "events_md", "describe"]
