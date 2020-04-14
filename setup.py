import os
from setuptools import setup, find_packages
from koelectra import __version__, __author__, __description__

def get_requires():
    result = []
    with open("./requirements.txt", "r") as f:
        for package_name in f:
            result.append(package_name)
    return result

setup(
    name='koelectra',                      # 모듈명
    version=__version__,             # 버전
    author=__author__,               # 저자
    description=__description__,     # 설명
    packages=find_packages(),
    python_requires='>=3.6.0',
    install_requires=get_requires(), # 패키지 설치를 위한 요구사항 패키지들
    include_package_data=True,
    zip_safe=False,
)
