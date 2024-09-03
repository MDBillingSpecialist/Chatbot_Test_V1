from setuptools import setup, find_packages

setup(
    name="my_project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pdfplumber",
        "fuzzywuzzy",
        "openai",
        "python-dotenv",
        "pytest",
        "rich"
    ],
    entry_points={
        "console_scripts": [
            "my_project=src.integration:main",
        ],
    },
)
