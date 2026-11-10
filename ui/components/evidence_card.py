from nicegui import ui

from ui.components.provenance_badge import provenance_badge


def evidence_card(item: dict) -> None:
    with ui.expansion(f"{item['document_id']} · p.{item.get('page') or '—'} · score {item['score']:.2f}", icon="article").classes("w-full"):
        ui.label(item["quote"]).classes("text-sm")
        ui.label(f"chunk_id: {item['chunk_id']} · source: {item['source_type']}").classes("text-xs text-grey-7")
        if item.get("needs_confirmation"):
            ui.label("需要工程师确认").props("color=warning icon=rule")
        provenance_badge("simulated")
