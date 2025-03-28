from ensure import ensure_annotations
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads yaml file and argument
    Args:
        path_to_yaml(str): path like input
    Raises :
       Value error : if yaml file is empty
       e: empty file

    Returns:
       configBox: ConfigBox type

    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file : {path_to_yaml} loaded sucessfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e