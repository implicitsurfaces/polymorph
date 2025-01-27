#!/usr/bin/env python3

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict


def style_str_to_dict(style_str: str) -> Dict[str, str]:
    """
    Converts an SVG style string into a dictionary.

    Args:
        style_str: Style string (e.g., "fill:none;stroke:black;stroke-width:2")

    Returns:
        Dictionary of style properties
    """
    if not style_str:
        return {}

    # Split the style string into key-value pairs
    style_pairs = style_str.split(";")

    # Convert to dictionary
    style_dict = {}
    for pair in style_pairs:
        if ":" in pair:
            key, value = pair.split(":", 1)
            style_dict[key.strip()] = value.strip()

    return style_dict


def dict_to_style_str(dict: Dict[str, str]) -> str:
    """
    Converts a dictionary into a SVG style string.

    Args:
        dict: Dictionary of style properties

    Returns:
        Style string
    """
    if not dict:
        return ""

    return ";".join(f"{key}:{value}" for key, value in dict.items())


def remove_ids(
    element,  #: Union[ET.Element, ET.ElementTree],
    recursive: bool = True,
) -> None:
    """
    Removes the 'id' attributes of elements.

    Args:
        element: ElementTree or Element to modify
        recursive: Whether to process child elements recursively
    """
    # If we're passed an ElementTree, get its root
    if isinstance(element, ET.ElementTree):
        element = element.getroot()

    # Remove id attribute if any
    if "id" in element.attrib:
        del element.attrib["id"]

    # Recursively process children if requested
    if recursive:
        for child in element:
            remove_ids(child, recursive)


def remove_style_properties(
    element,  #: Union[ET.Element, ET.ElementTree],
    properties: Dict[str, str | None],
    recursive: bool = True,
) -> None:
    """
    Removes the given style properties from an SVG element's style attribute.
    Also removes all stroke-related style properties if "stroke" is "none".

    Args:
        element: ElementTree or Element to modify
        properties: The key/value pairs of style properties to remove
        recursive: Whether to process child elements recursively
    """
    # If we're passed an ElementTree, get its root
    if isinstance(element, ET.ElementTree):
        element = element.getroot()

    # Process current element
    style = element.get("style", "")
    if style:
        # Parse current style attribute
        style_dict = style_str_to_dict(style)

        # Remove undesired key/value properties
        for property_name, property_value in properties.items():
            if property_name in style_dict and (
                property_value is None or style_dict[property_name] == property_value
            ):
                del style_dict[property_name]

        # Remove stroke attributes if no stroke
        if "stroke" not in style_dict or style_dict["stroke"] == "none":
            for property_name in [
                "stroke",
                "stroke-width",
                "stroke-linejoin",
                "stroke-linecap",
                "stroke-opacity",
                "stroke-dashoffset",
                "stroke-miterlimit",
            ]:
                if property_name in style_dict:
                    del style_dict[property_name]

        # Update or remove style attribute
        if style_dict:
            element.set("style", dict_to_style_str(style_dict))
        else:
            del element.attrib["style"]

    # Recursively process children if requested
    if recursive:
        for child in element:
            remove_style_properties(child, properties, recursive)


def remove_namespaced_attributes(
    element,  #: Union[ET.Element, ET.ElementTree]
) -> None:
    """
    Recursively removes all attributes that have a namespace from an ElementTree or Element.
    Modifies the element in place.

    Args:
        element: Either an ElementTree or Element object to clean
    """
    # If we're passed an ElementTree, get its root
    if isinstance(element, ET.ElementTree):
        element = element.getroot()

    # Get all attributes with namespaces
    namespaced_attrs = [attr for attr in element.attrib if "{" in attr or ":" in attr]

    # Remove them from the current element
    for attr in namespaced_attrs:
        del element.attrib[attr]

    # Recursively process all children
    for child in element:
        remove_namespaced_attributes(child)


def split_svg_group(input_file: Path, group_name: str, output_dir: Path):
    """
    Splits elements from a specified group in an SVG file into separate SVG files.

    Args:
        input_file (str): Path to the input SVG file
        group_name (str): Name of the group to split
        output_dir (str): Directory to save the output files
    """
    # Register the SVG namespace
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    # Parse the SVG file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Create output directory if it doesn't exist
    # We do this after parsing the SVG file so that it isn't
    # created if parsing fails, e.g., due to file not found.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the group
    label = "{http://www.inkscape.org/namespaces/inkscape}label"
    group = None
    for element in root.iter():
        if label in element.attrib and element.attrib[label] == group_name:
            group = element
            break

    if group is None:
        raise ValueError(f"Group '{group_name}' not found in the SVG file")

    # Process each element in the group
    for icon in group:
        # Get icon name
        if label not in icon.attrib:
            continue
        icon_name = icon.attrib[label]

        # Transfer all elements of the icon into a new SVG file.
        #
        # Note that it's important not to transfer the icon group itself.
        # Indeed the group contains a transform attribute to place it in the
        # toolbar, which we do not want in the separate icon file.
        #
        new_root = ET.Element(
            "svg", {"viewBox": "0 0 32 32", "width": "32px", "height": "32px"}
        )

        for element in icon:
            # Don't transfer the "IconBounds". This is an invisible rectangle
            # that gives the icon group a 32x32 bounding box, making it easy
            # to select and move in the source file, but we don't want it in
            # the generated separate icon file.
            if label in element.attrib and element.attrib[label] == "IconBounds":
                continue
            new_root.append(element)

        new_tree = ET.ElementTree(new_root)

        # Remove sodipodi/inkscape stuff
        remove_namespaced_attributes(new_tree)

        # Remove superfluous style properties
        remove_style_properties(
            new_tree,
            {
                "font-variation-settings": "normal",
                "opacity": "1",
                "fill-opacity": "1",
                "fill-rule": None,
                "stroke-linecap": "butt",
                "stroke-linejoin": "miter",
                "stroke-miterlimit": "4",
                "stroke-dasharray": "none",
                "stroke-dashoffset": "0",
                "stroke-opacity": "1",
                "paint-order": None,
                "stop-color": "#000000",
                "stop-opacity": "1",
                "vector-effect": None,
                "-inkscape-stroke": None,
                "display": "inline",
            },
        )

        # Remove IDs
        remove_ids(new_tree)

        # Save the new SVG file
        ET.indent(new_tree, space="  ")
        output_file = output_dir / f"{icon_name}.svg"
        new_tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print(f"Created: {output_file.absolute()}")


def main():
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    assets_dir = script_dir.parent / "src" / "assets"
    split_svg_group(script_dir / "tool-icons.svg", "Tools", assets_dir / "tool-icons")


if __name__ == "__main__":
    main()
