"""A small JSON-schema subset validator.

Supports ``type``, ``required``, ``properties``, ``items``, ``enum``,
``minLength``, ``maxLength``, ``pattern``, ``minimum`` and ``maximum``. Errors are
returned as readable strings that name the offending path (``$.foo`` for properties, ``$.foo[0]`` for array elements).
- type: one of string, number, integer, boolean, array, object, null
- required: list of required property names (objects)
- properties: property schemas (objects)
- enum: list of allowed values (any type)
- minimum / maximum: numeric bounds
- minLength / maxLength: string length bounds
- pattern: regular expression the string must match (re.search semantics)
- items: schema applied to each array element

Unknown keywords are ignored. Validation returns a list of human-readable
error strings, each annotated with a JSON path (``$`` for the root, ``$.foo``
for properties, ``$.foo[0]`` for array elements).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

__all__ = ["validate"]

_VALID_TYPES = (
    "string",
    "number",
    "integer",
    "boolean",
    "array",
    "object",
    "null",
)


def _type_name(value: Any) -> str:
    """Return the JSON type name for a Python value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _matches_type(value: Any, expected: str) -> bool:
    """Check whether a value conforms to a JSON type.

Args:
    value: The value to test.
    expected: JSON type name; ``number`` accepts ints and floats.

Booleans are never treated as numbers or integers, matching JSON Schema
semantics where ``true`` is of type boolean only.

    Booleans are never treated as numbers or integers, matching JSON Schema
    semantics where ``true`` is of type boolean only.
    """
    actual = _type_name(value)
    if expected == "number":
        return actual in ("integer", "number")
    return actual == expected


def _check_string_constraints(value: str, schema: Dict[str, Any], path: str, errors: List[str]) -> None:
    """Apply minLength, maxLength and pattern constraints to a string."""
    if "minLength" in schema and len(value) < schema["minLength"]:
        errors.append(
            "%s: string length %d is less than minLength %d"
            % (path, len(value), schema["minLength"])
        )
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        errors.append(
            "%s: string length %d is greater than maxLength %d"
            % (path, len(value), schema["maxLength"])
        )
    pattern = schema.get("pattern")
    if pattern is not None:
        try:
            if re.search(pattern, value) is None:
                errors.append("%s: string does not match pattern %r" % (path, pattern))
        except re.error as exc:
            errors.append("%s: invalid pattern %r in schema (%s)" % (path, pattern, exc))


def _check_number_constraints(value: Any, schema: Dict[str, Any], path: str, errors: List[str]) -> None:
    """Apply minimum and maximum constraints to a numeric value."""
    minimum = schema.get("minimum")
    if minimum is not None and value < minimum:
        errors.append("%s: value %s is less than minimum %s" % (path, value, minimum))
    maximum = schema.get("maximum")
    if maximum is not None and value > maximum:
        errors.append("%s: value %s is greater than maximum %s" % (path, value, maximum))


def _check_enum(value: Any, schema: Dict[str, Any], path: str, errors: List[str]) -> None:
    """Apply the enum constraint (list of allowed values)."""
    allowed = schema.get("enum")
    if allowed is not None and value not in allowed:
        errors.append(
            "%s: value %r is not one of the allowed values %r" % (path, value, allowed)
        )


def _validate_value(value: Any, schema: Dict[str, Any], path: str, errors: List[str]) -> None:
    """Recursively validate *value* against *schema*, appending errors."""
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        errors.append(
            "%s: expected type %r but got %r" % (path, expected_type, _type_name(value))
        )
        # Type mismatch makes nested checks meaningless; return early.
        return

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append("%s: missing required property %r" % (path, required))
        for key, sub_value in value.items():
            key_schema = properties.get(key)
            if key_schema is None:
                continue
            _validate_value(sub_value, key_schema, "%s.%s" % (path, key), errors)
    elif isinstance(value, list):
        items_schema = schema.get("items")
        if items_schema is not None:
            for index, element in enumerate(value):
                _validate_value(element, items_schema, "%s[%d]" % (path, index), errors)
    elif isinstance(value, str):
        _check_string_constraints(value, schema, path, errors)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _check_number_constraints(value, schema, path, errors)

    _check_enum(value, schema, path, errors)


def validate(instance: Any, schema: Dict[str, Any]) -> List[str]:
    """Validate an instance against a schema.

    Args:
        instance: The value to validate.
        schema: The schema to validate it against.

    Returns:
        A list of error messages. Empty list means the instance is valid.
    """
    errors: List[str] = []
    _validate_value(instance, schema, "$", errors)
    return errors
