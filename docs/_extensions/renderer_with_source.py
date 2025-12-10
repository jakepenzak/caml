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
        header = super().render_header(el) # pyright: ignore[reportArgumentType]
        if hasattr(el.obj, "source_link") and el.obj.source_link:
            split_header = header.split("{")
            source_link = f'[↗]({el.obj.source_link} "View Source Code")'
            result = split_header[0] + source_link + "{" + split_header[1]
            return result
        else:
            return header


    @dispatch
    def signature(self, el: layout.Doc) -> str:
        sig = super().signature(el) # pyright: ignore[reportArgumentType]
        if hasattr(el.obj, "source_link") and el.obj.source_link:
            source_link = f'\n\n<div style="text-align: right;"> [View Source Code]({el.obj.source_link} "View Source Code") </div>'
            result = sig + source_link
            return result
        else:
            return sig
