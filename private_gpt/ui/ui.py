"""This file should be imported only and only if you want to run the UI locally."""
import itertools
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from llama_index.llms import ChatMessage, ChatResponse, MessageRole

from private_gpt.di import root_injector
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.chunks.chunks_service import ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg

logger = logging.getLogger(__name__)


UI_TAB_TITLE = "My Private GPT"
SERUM_GUIDES = {
    "reese bass": {
        "title": "Reese Bass",
        "subtitle": "Classic detuned saw bass with motion and grit.",
        "steps": [
            {
                "title": "Start from Init",
                "details": "Menu → Init Preset to clear modulation and FX.",
                "settings": ["Voices: 1", "Master: 0 dB"],
                "visual": "power",
            },
            {
                "title": "Osc A: Wide Saw",
                "details": "Use a saw wave and widen with unison.",
                "settings": ["Wave: Saw", "Unison: 7", "Detune: 0.12"],
                "visual": "saw",
            },
            {
                "title": "Osc B: Lower Saw",
                "details": "Blend a lower saw to thicken the low mids.",
                "settings": ["Wave: Saw", "Oct: -1", "Unison: 4"],
                "visual": "saw_low",
            },
            {
                "title": "Filter: Low Pass",
                "details": "Tame highs and emphasize growl.",
                "settings": ["MG Low 12", "Cutoff: 180 Hz", "Drive: 10%"],
                "visual": "filter",
            },
            {
                "title": "Envelope Shape",
                "details": "Punchy transient with a medium tail.",
                "settings": ["A: 2 ms", "D: 450 ms", "S: 40%", "R: 120 ms"],
                "visual": "env",
            },
            {
                "title": "LFO Wobble",
                "details": "Add movement by modulating cutoff.",
                "settings": ["LFO: 1/8", "Shape: Triangle", "Cutoff: ±20%"],
                "visual": "lfo",
            },
            {
                "title": "FX: Dirt + Glue",
                "details": "Distortion and compression for weight.",
                "settings": [
                    "Distortion: Diode 30%",
                    "Chorus: Mix 20%",
                    "Comp: Multiband 40%",
                ],
                "visual": "fx",
            },
            {
                "title": "Sub Support",
                "details": "Enable a sine sub to anchor the low end.",
                "settings": ["Sub Osc: Sine", "Oct: -1", "Level: 45%"],
                "visual": "sub",
            },
        ],
        "tips": [
            "Stack slight detune differences between A and B for stereo width.",
            "Automate LFO rate for build-up energy.",
            "EQ a small notch around 250 Hz to reduce boxiness.",
        ],
    }
}


class PrivateGptUi:
    def __init__(self) -> None:
        self._ingest_service = root_injector.get(IngestService)
        self._chat_service = root_injector.get(ChatService)
        self._chunks_service = root_injector.get(ChunksService)

        # Cache the UI blocks
        self._ui_block = None

    def _chat(self, message: str, history: list[list[str]], mode: str, *_: Any) -> Any:
        def yield_deltas(stream: Iterable[ChatResponse | str]) -> Iterable[str]:
            full_response: str = ""
            for delta in stream:
                if isinstance(delta, str):
                    full_response += str(delta)
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                yield full_response

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = list(
                itertools.chain(
                    *[
                        [
                            ChatMessage(content=interaction[0], role=MessageRole.USER),
                            ChatMessage(
                                content=interaction[1], role=MessageRole.ASSISTANT
                            ),
                        ]
                        for interaction in history
                    ]
                )
            )

            # max 20 messages to try to avoid context overflow
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]
        match mode:
            case "Query Docs":
                query_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                )
                yield from yield_deltas(query_stream)

            case "LLM Chat":
                llm_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=False,
                )
                yield from yield_deltas(llm_stream)

            case "Search in Docs":
                response = self._chunks_service.retrieve_relevant(
                    text=message, limit=4, prev_next_chunks=0
                )

                yield "\n\n\n".join(
                    f"{index}. **{chunk.document.doc_metadata['file_name'] if chunk.document.doc_metadata else ''} "
                    f"(page {chunk.document.doc_metadata['page_label'] if chunk.document.doc_metadata else ''})**\n "
                    f"{chunk.text}"
                    for index, chunk in enumerate(response, start=1)
                )

    def _list_ingested_files(self) -> list[list[str]]:
        files = set()
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.doc_metadata is None:
                # Skipping documents without metadata
                continue
            file_name = ingested_document.doc_metadata.get(
                "file_name", "[FILE NAME MISSING]"
            )
            files.add(file_name)
        return [[row] for row in files]

    def _upload_file(self, file: TextIO) -> None:
        path = Path(file.name)
        self._ingest_service.ingest(file_name=path.name, file_data=path)

    def _render_serum_guide(self, sound_name: str) -> str:
        normalized_name = sound_name.strip().lower()
        guide = SERUM_GUIDES.get(normalized_name)
        if guide is None:
            options = ", ".join(f"<code>{name}</code>" for name in SERUM_GUIDES)
            return (
                "<div class='serum-empty'>"
                "<h3>No guide found yet</h3>"
                "<p>Try one of the available sounds:</p>"
                f"<p>{options}</p>"
                "</div>"
            )

        steps_html = "".join(
            self._render_serum_step(step, index + 1)
            for index, step in enumerate(guide["steps"])
        )
        tips_html = "".join(f"<li>{tip}</li>" for tip in guide["tips"])
        return (
            "<section class='serum-guide'>"
            "<header class='serum-header'>"
            f"<h2>{guide['title']}</h2>"
            f"<p>{guide['subtitle']}</p>"
            "</header>"
            f"<div class='serum-steps'>{steps_html}</div>"
            "<aside class='serum-tips'>"
            "<h3>Quick Tips</h3>"
            f"<ul>{tips_html}</ul>"
            "</aside>"
            "</section>"
        )

    def _render_serum_step(self, step: dict[str, Any], number: int) -> str:
        settings_html = "".join(f"<li>{setting}</li>" for setting in step["settings"])
        return (
            "<article class='serum-step'>"
            "<div class='serum-step-top'>"
            f"<div class='serum-step-number'>{number:02d}</div>"
            f"{self._render_serum_visual(step['visual'])}"
            "</div>"
            "<div class='serum-step-body'>"
            f"<h4>{step['title']}</h4>"
            f"<p>{step['details']}</p>"
            f"<ul>{settings_html}</ul>"
            "</div>"
            "</article>"
        )

    def _render_serum_visual(self, visual: str) -> str:
        visuals = {
            "power": """
                <svg viewBox="0 0 64 64" role="img" aria-label="Init preset">
                    <circle cx="32" cy="32" r="22" class="serum-svg-ring"/>
                    <rect x="30" y="10" width="4" height="20" rx="2" class="serum-svg-fill"/>
                </svg>
            """,
            "saw": """
                <svg viewBox="0 0 64 64" role="img" aria-label="Saw wave">
                    <path d="M10 48 L26 16 L26 48 L42 16 L42 48 L54 24" class="serum-svg-stroke"/>
                </svg>
            """,
            "saw_low": """
                <svg viewBox="0 0 64 64" role="img" aria-label="Lower saw wave">
                    <path d="M12 44 L28 20 L28 44 L44 20 L44 44 L52 30" class="serum-svg-stroke"/>
                    <circle cx="12" cy="52" r="3" class="serum-svg-fill"/>
                    <circle cx="28" cy="52" r="3" class="serum-svg-fill"/>
                </svg>
            """,
            "filter": """
                <svg viewBox="0 0 64 64" role="img" aria-label="Filter">
                    <path d="M12 18 H52" class="serum-svg-stroke"/>
                    <path d="M12 34 H44" class="serum-svg-stroke"/>
                    <path d="M12 50 H36" class="serum-svg-stroke"/>
                    <circle cx="48" cy="34" r="5" class="serum-svg-fill"/>
                </svg>
            """,
            "env": """
                <svg viewBox="0 0 64 64" role="img" aria-label="Envelope">
                    <path d="M10 50 L18 18 L38 32 L54 26" class="serum-svg-stroke"/>
                    <circle cx="18" cy="18" r="3" class="serum-svg-fill"/>
                </svg>
            """,
            "lfo": """
                <svg viewBox="0 0 64 64" role="img" aria-label="LFO">
                    <path d="M8 40 Q20 16 32 40 T56 40" class="serum-svg-stroke"/>
                </svg>
            """,
            "fx": """
                <svg viewBox="0 0 64 64" role="img" aria-label="FX">
                    <rect x="12" y="14" width="40" height="36" rx="8" class="serum-svg-ring"/>
                    <circle cx="24" cy="32" r="6" class="serum-svg-fill"/>
                    <circle cx="40" cy="32" r="6" class="serum-svg-fill"/>
                </svg>
            """,
            "sub": """
                <svg viewBox="0 0 64 64" role="img" aria-label="Sub oscillator">
                    <circle cx="32" cy="32" r="18" class="serum-svg-ring"/>
                    <path d="M20 32 H44" class="serum-svg-stroke"/>
                </svg>
            """,
        }
        return f"<div class='serum-visual'>{visuals.get(visual, '')}</div>"

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(primary_hue=slate),
            css=".logo { "
            "display:flex;"
            "background-color: #C7BAFF;"
            "height: 80px;"
            "border-radius: 8px;"
            "align-content: center;"
            "justify-content: center;"
            "align-items: center;"
            "}"
            ".logo img { height: 25% }"
            ".serum-guide {"
            "display: flex;"
            "flex-direction: column;"
            "gap: 24px;"
            "padding: 16px;"
            "background: #f7f5ff;"
            "border-radius: 16px;"
            "}"
            ".serum-header h2 {"
            "margin-bottom: 4px;"
            "}"
            ".serum-steps {"
            "display: grid;"
            "grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));"
            "gap: 16px;"
            "}"
            ".serum-step {"
            "background: #ffffff;"
            "border-radius: 14px;"
            "padding: 16px;"
            "box-shadow: 0 8px 20px rgba(23, 12, 61, 0.08);"
            "display: flex;"
            "flex-direction: column;"
            "gap: 12px;"
            "}"
            ".serum-step-top {"
            "display: flex;"
            "align-items: center;"
            "justify-content: space-between;"
            "}"
            ".serum-step-number {"
            "font-size: 18px;"
            "font-weight: 700;"
            "color: #6d5cff;"
            "background: rgba(109, 92, 255, 0.12);"
            "padding: 6px 10px;"
            "border-radius: 999px;"
            "}"
            ".serum-visual {"
            "width: 64px;"
            "height: 64px;"
            "display: flex;"
            "align-items: center;"
            "justify-content: center;"
            "background: #efeaff;"
            "border-radius: 12px;"
            "}"
            ".serum-visual svg {"
            "width: 44px;"
            "height: 44px;"
            "}"
            ".serum-svg-stroke {"
            "fill: none;"
            "stroke: #6d5cff;"
            "stroke-width: 4;"
            "stroke-linecap: round;"
            "stroke-linejoin: round;"
            "}"
            ".serum-svg-fill {"
            "fill: #6d5cff;"
            "}"
            ".serum-svg-ring {"
            "fill: none;"
            "stroke: #6d5cff;"
            "stroke-width: 4;"
            "}"
            ".serum-step-body h4 {"
            "margin-bottom: 4px;"
            "}"
            ".serum-step-body ul {"
            "margin: 0;"
            "padding-left: 18px;"
            "color: #463c6f;"
            "}"
            ".serum-tips {"
            "background: #ffffff;"
            "border-radius: 12px;"
            "padding: 16px;"
            "}"
            ".serum-tips ul {"
            "margin: 0;"
            "padding-left: 18px;"
            "}"
            ".serum-empty {"
            "padding: 24px;"
            "border-radius: 12px;"
            "background: #fff5f5;"
            "}",
        ) as blocks:
            with gr.Row():
                gr.HTML(f"<div class='logo'/><img src={logo_svg} alt=PrivateGPT></div")

            with gr.Tabs():
                with gr.Tab(label="Chat"):
                    with gr.Row():
                        with gr.Column(scale=3, variant="compact"):
                            mode = gr.Radio(
                                ["Query Docs", "Search in Docs", "LLM Chat"],
                                label="Mode",
                                value="Query Docs",
                            )
                            upload_button = gr.components.UploadButton(
                                "Upload a File",
                                type="file",
                                file_count="single",
                                size="sm",
                            )
                            ingested_dataset = gr.List(
                                self._list_ingested_files,
                                headers=["File name"],
                                label="Ingested Files",
                                interactive=False,
                                render=False,  # Rendered under the button
                            )
                            upload_button.upload(
                                self._upload_file,
                                inputs=upload_button,
                                outputs=ingested_dataset,
                            )
                            ingested_dataset.change(
                                self._list_ingested_files,
                                outputs=ingested_dataset,
                            )
                            ingested_dataset.render()
                        with gr.Column(scale=7):
                            _ = gr.ChatInterface(
                                self._chat,
                                chatbot=gr.Chatbot(
                                    label=f"LLM: {settings.llm.mode}",
                                    show_copy_button=True,
                                    render=False,
                                    avatar_images=(
                                        None,
                                        "https://lh3.googleusercontent.com/drive-viewer/"
                                        "AK7aPaAicXck0k68nsscyfKrb18o9ak3BSaWM_"
                                        "Qzm338cKoQlw72Bp0UKN84IFZjXjZApY01mtnUX"
                                        "DeL4qzwhkALoe_53AhwCg=s2560",
                                    ),
                                ),
                                additional_inputs=[mode, upload_button],
                            )
                with gr.Tab(label="Serum 2 Guide"):
                    with gr.Column():
                        gr.Markdown(
                            "### Build a sound in **Serum 2**\n"
                            "Search for a sound (e.g. `Reese bass`) to get a step-by-step visual guide."
                        )
                        with gr.Row():
                            sound_input = gr.Dropdown(
                                label="Sound search",
                                choices=sorted(SERUM_GUIDES.keys()),
                                value="reese bass",
                                allow_custom_value=True,
                                filterable=True,
                                scale=3,
                            )
                            build_button = gr.Button(
                                "Generate Guide", variant="primary"
                            )
                        guide_output = gr.HTML(
                            value=self._render_serum_guide("Reese bass")
                        )
                        build_button.click(
                            self._render_serum_guide,
                            inputs=sound_input,
                            outputs=guide_output,
                        )
        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        base_path = settings.ui.path
        logger.info("Mounting the gradio UI, at path=%s", base_path)
        gr.mount_gradio_app(app, blocks, path=base_path)


if __name__ == "__main__":
    ui = PrivateGptUi()
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)
