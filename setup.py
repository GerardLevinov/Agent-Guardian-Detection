"""
Setup configuration for Agent Guardian Detector
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = ""
readme_file = this_directory / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

setup(
    name="agent-guardian-detector",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Policy enforcement system for CrewAI applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Agent-Guardian-Detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "litellm>=1.0.0",
        "crewai>=0.1.0",
        "pyyaml>=6.0",
        "ollama>=0.1.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)