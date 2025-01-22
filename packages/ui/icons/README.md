For convenience, icons are meant to be designed via a "master file" (e.g., `tool-icons.svg`) that contains several icons, making it easier to have a coherent style.

Then, individual icons can be generated via the `generate-icons.py` script that takes as input an SVG master file and the name of the group where icons are,

Example:

```
cd polymorph/packages/ui/icons
./generate-icons.py tool-icons.svg Tools
```
