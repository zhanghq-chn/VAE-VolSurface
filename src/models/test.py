from src.utils.yaml_helper import YamlParser

vae_dict = YamlParser("src/models/vae.yaml").load_yaml()
print(vae_dict)