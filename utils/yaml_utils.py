import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def update_config(config, key, value):
    keys = key.split('.')
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # 尝试转换值类型
    try:
        d[keys[-1]] = yaml.safe_load(value)
    except yaml.YAMLError:
        d[keys[-1]] = value

def load_config_with_options(config_path, options=None):
    config = load_config(config_path)
    if options:
        for opt in options:
            if '=' not in opt:
                raise ValueError(f"Invalid option: {opt}")
            key, value = opt.split('=', 1)
            update_config(config, key, value)
    return config
