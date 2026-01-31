from plum import dispatch
from quartodoc import layout
from quartodoc.renderers import MdRenderer


class Renderer(MdRenderer):
    """Markdown renderer with source code links.

    Extends the base MdRenderer to add a "View source code" link after
    each function/class header that points to the GitHub repository.
    """

    style = "markdown_with_source"

    @dispatch
    def render_header(self, el: layout.Doc) -> str:
        header = super().render_header(el)  # pyright: ignore[reportArgumentType]
        if hasattr(el.obj, "source_link") and el.obj.source_link:
            source_link = f'\n<small class="text-muted">↗[View Source Code]({el.obj.source_link})</small>'
            result = "<br>\n\n" + header + source_link
            return result
        else:
            return header
