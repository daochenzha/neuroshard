import setuptools

setuptools.setup(
    name="neuroshard",
    version='1.0.0',
    author="Daochen Zha",
    author_email="daochen.zha@rice.edu",
    description="neuroshard",
    url="https://github.com/daochenzha/neuroshard",
    keywords=["Sharding"],
    packages=setuptools.find_packages(exclude=('tests',)),
    requires_python='>=3.8',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
