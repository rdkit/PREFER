import os
import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="prefer",
    use_scm_version=True,
    license="MIT",
    author="Jessica Lanini",
    author_email="jessica.lanini@novartis.com",
    description="benchmarking and Property pREdiction FramEwoRk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://dev.azure.com/MSAI-DevOps-Org/FormulaOne%20Azure%20-%20AI%20Exploration%20-%20Gen%20Chem/_git/PREFER", #TO DO change it accprding to the final GitHub location
    setup_requires=["setuptools_scm"],
    python_requires="==3.7.7",
    install_requires=[
        "dpu-utils>=0.2.13",
        "scikit-learn==0.24.1",
        "numpy==1.19.2",
        "pandas>=1.2.4",
        "auto-sklearn==0.14.7",
    ],
    packages=setuptools.find_packages(),
    entry_points={"console_scripts": ["prefer = prefer.run_prefer_automation:run_PREFER"]},
)
