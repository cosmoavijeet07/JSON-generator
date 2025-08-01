from jsonschema import Draft7Validator, exceptions as jsonschema_exceptions

def is_valid_schema(schema_json):
    try:
        Draft7Validator.check_schema(schema_json)
        return True, None
    except jsonschema_exceptions.SchemaError as e:
        return False, str(e)
