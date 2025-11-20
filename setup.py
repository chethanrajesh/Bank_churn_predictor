from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bank-churn-prediction",
    version="1.0.0",
    author="Priyangkush Debnath",
    author_email="dpriyangkush004@gmail.com",
    description="A comprehensive machine learning system for bank customer churn prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DPriyangkush/Customer-Churn-Prediction-System-with-Explainable-AI.git",
    packages=find_packages(include=['src', 'src.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "ui": [
            "streamlit>=1.28.0",
            "plotly>=5.17.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "generate-churn-data=src.synthetic_data_generator:main",
            "train-churn-model=src.model_training:main",
            "predict-churn=src.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["../config/*.yaml", "../config/*.json"],
    },
)