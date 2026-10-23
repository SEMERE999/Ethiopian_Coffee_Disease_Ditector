from setuptools import find_packages, setup
from typing import List


def get_requirements() -> List[str]:
    """
    Reads requirements from requirements.txt, filtering out empty lines and editable flags.
    """
    requirements_list: List[str] = []
    try:
        with open("requirements.txt", "r", encoding="utf-8-sig") as file:
            lines = file.readlines()
            for line in lines:
                req = line.strip()
                if req and req != "-e ." and not req.startswith("#"):
                    requirements_list.append(req)
    except FileNotFoundError:
        print("requirements.txt not found")

    return requirements_list


setup(
    name="EthiopianCoffee",
    version="0.0.1",
    author="Semere Brhane",
    author_email="semrawiiii@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements(),
)
