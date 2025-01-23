For convenience, icons are meant to be designed via a "master file" (e.g., `tool-icons.svg`) that contains several icons, making it easier to have a coherent style.

Then, individual icons can be generated via the `generate-icons.py` script that takes as input an SVG master file and the name of the group where icons are,

Example:

```
cd polymorph/packages/ui/icons
./generate-icons.py tool-icons.svg Tools
```

## How to create a new tool icon?

1. Open `tool-icons.svg` with Inkscape and open the Layers panel (Ctrl+Maj+L).

2. In the "Tools" layer, duplicate one of the icon group.

3. Rename it to the desired icon name (the group name is used by `generate-icons.py`).

4. Translate the group to its appropriate location in the toolbar. You can enable
   snapping (shortcut: %) to make this easier and pixel-perfect.

> Warning:
>
> Make sure to translate the group itself and not the elements within the group.
> Indeed, this group translation will be discarded when the script generates the
> individual icon size, but if instead of moving the group you move its inner content,
> the script will assume that this translation is part of the icon itself.

5. Unlock the "Template" layer, duplicate one of the icon template, move it under the
   new icon (it is a guide to help you design the icon with the appropriate padding),
   and re-lock the layer.

6. Back in the "Tools" layer, double click in your new icon group (to "enter" the group)
   and edit the icon.

7. Save and run the script: `./generate-icons.py tool-icons.svg Tools`

Tips:

- It is useful to enable a 1px grid and enable snapping to the grid in order
  to get pixel-perfect lines. Although when you need half-pixels (for example,
  for the y-position of a horizontal 1px wide line), you'd typically
  have to type that manually.

- Saving and opening the SVG master file in a web browser is useful to get
  a 100% preview of how the icon looks.

- The XML editor (Ctrl+Maj+X) is useful to check that there is nothing
  extraneous that you do not want, and that the icon definition is clean.
  Don't worry about the super long "style" attribute added by Inkscape, this
  will be cleaned up automatically by the script.

- If you're using a stroke that doesn't care about having a specific linecap
  or linejoin (e.g., because these are hidden), use the default: "butt" and "miter".
  Since these are the default SVG values, it will be stripped out of the final
  file and therefore lead to a smaller file size.
