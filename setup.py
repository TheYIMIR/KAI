from setuptools import setup, find_packages

setup(
    name="kai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "nltk>=3.8.0",
        "wikipedia>=1.4.0",
        "requests>=2.30.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Knowledge Adaptive Intelligence - A self-learning AI system",
    keywords="ai, ml, knowledge, learning",
    python_requires=">=3.8",
)