# %%
import json
import yaml
import numpy as np
import datetime
import os
import shutil
from typing import Any

def create_directory(directory_name: str = "tmp", force: bool = False) -> None:
    """making new directory (name: default to "tmp")

    Args:
        directory_name (str, optional): directory name. Defaults to "tmp".
        force (bool, optional): make new directory and overwrite the existing directory even if the "directory_name" directory is exist. Defaults to False.
    """
    print(f"try to making <{directory_name}> directory.")

    if os.path.exists(directory_name):
        if force:
            shutil.rmtree(directory_name)
            print(f"Removed existing directory: {directory_name}")
        else:
            print(f"The directory <{directory_name}> already exists.")
            return

    os.makedirs(directory_name)
    print(f"Created a new directory: {directory_name}")


def save_str(path:str, string: str)->None:
    with open(path, "w") as file:
        file.write(string)

def make_dir_name(file_name:str="output")->str:
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{file_name}_{timestamp}"
    return filename

def get_base_name(str:str)->str:
    basename = os.path.basename(str)
    return basename.split(".")[0]

def instance_to_dict(instance: Any, properties_list: list[str])->dict:
    return {property: instance.__dict__[property] for property in properties_list}



if __name__ == "__main__":
    create_directory("tmp")
    dirname = make_dir_name()
    create_directory(f"tmp/{dirname}", force=True)
    print(get_base_name("result/output_2023-12-06-19-43-56/con_3330.npy"))

    class MyClass:
        def __init__(self, name:str, age:int, location:str):
            self.name = name
            self.age = age
            self.location = location

    my_instance = MyClass("John Doe", 30, "New York")
    instance_dict = instance_to_dict(my_instance, ["name", "age"])
    yaml_str = yaml.dump(instance_dict)
    save_str("tmp/test.yaml", yaml_str)


# %%
